import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cnn_networks import EncoderAttn, DecoderAttn
from model.vq.quantizer import MSQuantizer


class SegmentAttnPool(nn.Module):
    """CLIP text embedding을 query로 사용해 motion feature를 N개 token으로 압축"""
    def __init__(self, clip_dim, motion_dim, num_heads=8, dropout=0.1,
                 max_n_seg=8, max_motion_len=49):
        super().__init__()
        self.q_proj = nn.Linear(clip_dim, motion_dim)
        self.attn = nn.MultiheadAttention(motion_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(motion_dim)
        # segment 순서 PE: query_i가 i번째 segment임을 알려줌
        self.seg_pe = nn.Embedding(max_n_seg, motion_dim)
        # motion temporal PE: 각 frame의 시간 위치 알려줌
        self.motion_pe = nn.Embedding(max_motion_len, motion_dim)

    def forward(self, clip_embeds, motion_feats, seg_mask=None):
        # clip_embeds: [B, N, clip_dim]
        # motion_feats: [B, T, motion_dim]
        # seg_mask: [B, N] bool, True=valid segment
        B, N, _ = clip_embeds.shape
        T = motion_feats.shape[1]

        seg_pos = torch.arange(N, device=clip_embeds.device)
        Q = self.q_proj(clip_embeds) + self.seg_pe(seg_pos).unsqueeze(0)   # [B, N, motion_dim]

        mot_pos = torch.arange(T, device=motion_feats.device)
        KV = motion_feats + self.motion_pe(mot_pos).unsqueeze(0)            # [B, T, motion_dim]

        out, _ = self.attn(Q, KV, KV)
        if seg_mask is not None:
            out = out * seg_mask.unsqueeze(-1).float()
        return self.norm(out)


class SimpleVQ(nn.Module):
    """EMA 업데이트 VQ codebook (MSQuantizer 방식 data-driven 초기화)"""
    def __init__(self, nb_code=512, code_dim=512, mu=0.99):
        super().__init__()
        self.nb_code  = nb_code
        self.code_dim = code_dim
        self.mu       = mu
        self.register_buffer('codebook',  torch.zeros(nb_code, code_dim))
        self.register_buffer('code_sum',  torch.zeros(nb_code, code_dim))
        self.register_buffer('code_count', torch.zeros(nb_code))
        self.register_buffer('_inited',   torch.tensor(False))

    def _tile(self, x):
        n, d = x.shape
        if n < self.nb_code:
            n_rep = (self.nb_code + n - 1) // n
            out   = x.repeat(n_rep, 1)[:self.nb_code]
            out   = out + torch.randn_like(out) * (0.01 / d ** 0.5)  # tiny noise to break ties
        else:
            out = x
        return out[:self.nb_code]

    @torch.no_grad()
    def _init_codebook(self, x):
        """첫 배치 실제 데이터로 codebook 초기화 (MSQuantizer 동일 방식)"""
        out = self._tile(x.float())
        self.codebook[:]   = out
        self.code_sum[:]   = out.clone()
        self.code_count[:] = 1.0   # all codes start alive (MSQuantizer 동일)
        self._inited.fill_(True)

    def quantize(self, x):
        dist = torch.cdist(x.float(), self.codebook.float())
        return dist.argmin(dim=-1)

    @torch.no_grad()
    def _ema_update(self, x, code_idx):
        one_hot = F.one_hot(code_idx, self.nb_code).float()
        self.code_count = self.mu * self.code_count + (1 - self.mu) * one_hot.sum(0)
        self.code_sum   = self.mu * self.code_sum   + (1 - self.mu) * (one_hot.T @ x.float())

        # alive codes: EMA update
        alive = self.code_count > 1e-3
        self.codebook[alive] = (
            self.code_sum[alive] / self.code_count[alive].unsqueeze(-1).clamp(min=1e-6)
        )
        # dead codes: reset to a random data vector
        dead = ~alive
        if dead.any():
            rand_idx = torch.randint(len(x), (dead.sum().item(),), device=x.device)
            self.codebook[dead] = x[rand_idx].float()

    def forward(self, x, seg_mask=None):
        # x: [B, N, D]
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        valid  = (seg_mask.reshape(B * N).bool()
                  if seg_mask is not None
                  else torch.ones(B * N, dtype=torch.bool, device=x.device))

        # data-driven init on first training batch (before first quantize)
        if self.training and not self._inited.item():
            self._init_codebook(x_flat[valid])

        code_idx = self.quantize(x_flat)
        x_q = self.codebook[code_idx].reshape(B, N, D)
        x_q_st = x + (x_q - x).detach()   # straight-through

        commit_loss = F.mse_loss(x_q.detach(), x, reduction='none').mean(-1)  # [B, N]
        if seg_mask is not None:
            commit_loss = (commit_loss * seg_mask.float()).sum() / seg_mask.float().sum().clamp(min=1)
        else:
            commit_loss = commit_loss.mean()

        # codebook usage stats (collapse indicator)
        # perplexity: exp(H) → 512=perfect uniform, ~1=collapsed
        with torch.no_grad():
            avg_probs  = F.one_hot(code_idx, self.nb_code).float().mean(0)  # [V]
            perplexity = (-(avg_probs * (avg_probs + 1e-10).log()).sum()).exp()
            n_active   = (avg_probs > 0).sum()

        if self.training:
            self._ema_update(x_flat[valid], code_idx[valid])

        return x_q_st, commit_loss, code_idx.reshape(B, N), perplexity, n_active


class SegVQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        motion_dim   = cfg.model.motion_dim      # 263
        latent_dim   = cfg.model.latent_dim      # 512
        down_t       = cfg.model.down_t          # 2 (→ 4x downsample)
        stride_t     = cfg.model.stride_t        # 2
        width        = cfg.model.width           # 512
        depth        = cfg.model.depth           # 3
        dil          = cfg.model.dilation_growth_rate  # 3
        clip_dim     = cfg.model.clip_dim        # 512
        nb_code      = cfg.model.nb_code         # 512
        self.down_t  = down_t
        self.lambda_align  = cfg.training.lambda_align
        self.lambda_commit = cfg.training.lambda_commit

        self.encoder = EncoderAttn(motion_dim, latent_dim, down_t, stride_t,
                                   width, depth, dil, activation='relu', use_attn=True)
        self.decoder = DecoderAttn(motion_dim, latent_dim, down_t, stride_t,
                                   width, depth, dil, activation='relu', use_attn=True)
        max_n_seg    = cfg.data.max_n_seg if hasattr(cfg, 'data') else 8
        max_motion_len = cfg.model.get('max_motion_len', 49)
        self.seg_pool = SegmentAttnPool(clip_dim, latent_dim,
                                        max_n_seg=max_n_seg,
                                        max_motion_len=max_motion_len)
        self.vq = SimpleVQ(nb_code, latent_dim)

        # scale1~4: HRVQVAE와 동일한 MSQuantizer (residual 처리)
        self.quantizer = MSQuantizer(
            nb_code=nb_code,
            code_dim=latent_dim,
            mu=cfg.model.get('mu', 0.99),
            scales=cfg.model.get('scales', [8, 4, 2, 1]),
            share_quant_resi=cfg.model.get('share_quant_resi', 4),
            quant_resi=cfg.model.get('quant_resi', 0.5),
        )
        self.lambda_commit_hrv = cfg.training.get('lambda_commit_hrv', 0.02)

        self.align_proj_motion = nn.Linear(latent_dim, latent_dim)
        self.align_proj_text   = nn.Linear(clip_dim, latent_dim)

    def _temporal_seg_pool(self, feat, seg_mask, m_lens_down):
        """
        encoder feature를 segment 수로 equal split해 avg pool.
        boundary 정보 없이 n_valid등분 → SimpleVQ/SegmentAttnPool과 완전히 독립.

        feat:        [B, T, D]
        seg_mask:    [B, N] bool
        m_lens_down: [B]  실제 downsampled 길이
        returns:     [B, N, D]
        """
        B, T, D = feat.shape
        N = seg_mask.shape[1]
        out = torch.zeros(B, N, D, device=feat.device)

        for b in range(B):
            n_valid = int(seg_mask[b].sum().item())
            if n_valid == 0:
                continue
            length = int(m_lens_down[b].item()) if m_lens_down is not None else T
            length = min(max(length, 1), T)
            edges  = torch.linspace(0, length, n_valid + 1).long()
            for i in range(n_valid):
                t0 = edges[i].item()
                t1 = max(edges[i + 1].item(), t0 + 1)
                out[b, i] = feat[b, t0:min(t1, T)].mean(0)

        return out  # [B, N, D]

    def _seg_align_loss(self, seg_feats, clip_embeds, seg_mask, temperature=0.07):
        """
        Bidirectional InfoNCE alignment loss (CLIP 방식).

        margin 기반 triplet과 달리 temperature로 sharpening → gradient가 항상 흐름.
        seg_feats:   [B, N, latent_dim]
        clip_embeds: [B, N, clip_dim]
        seg_mask:    [B, N] bool
        """
        B, N, _ = seg_feats.shape

        m_proj = F.normalize(self.align_proj_motion(seg_feats), dim=-1)   # [B, N, A]
        t_proj = F.normalize(self.align_proj_text(clip_embeds), dim=-1)   # [B, N, A]

        flat_mask = (seg_mask.reshape(B * N) if seg_mask is not None
                     else torch.ones(B * N, dtype=torch.bool, device=seg_feats.device))
        vm = m_proj.reshape(B * N, -1)[flat_mask]   # [V, A]
        vt = t_proj.reshape(B * N, -1)[flat_mask]   # [V, A]
        V  = vm.shape[0]
        if V < 2:
            return vm.sum() * 0.

        sim    = vm @ vt.T / temperature             # [V, V]
        labels = torch.arange(V, device=vm.device)

        loss_mt = F.cross_entropy(sim,   labels)     # m_i → 정답 t_i
        loss_tm = F.cross_entropy(sim.T, labels)     # t_i → 정답 m_i

        return (loss_mt + loss_tm) * 0.5

        # ── 기존 Bidirectional hard-negative triplet loss (margin=0.2) ──────────
        # margin=0.2 를 넘으면 gradient가 0이 되어 positive를 더 이상 당기지 않음
        # → cos_sim std가 낮게 수렴하는 원인 → InfoNCE로 교체
        #
        # sim = vm @ vt.T
        # d_pos   = 1. - sim.diagonal()
        # sim_mt  = sim.clone();  sim_mt.fill_diagonal_(-2.)
        # d_neg_t = 1. - sim_mt.max(dim=1).values
        # sim_tm  = sim.T.clone(); sim_tm.fill_diagonal_(-2.)
        # d_neg_m = 1. - sim_tm.max(dim=1).values
        # loss_mt = torch.clamp(d_pos - d_neg_t + margin, min=0.).mean()
        # loss_tm = torch.clamp(d_pos - d_neg_m + margin, min=0.).mean()
        # return (loss_mt + loss_tm) * 0.5
        # ─────────────────────────────────────────────────────────────────────────

    def encode(self, motion, m_lens=None):
        # motion: [B, T, 263]
        x = motion.permute(0, 2, 1).float()      # [B, 263, T]
        feat = self.encoder(x, m_lens)            # [B, latent_dim, T/4]
        return feat.permute(0, 2, 1)             # [B, T/4, latent_dim]

    def forward(self, motion, clip_embeds, seg_mask, m_lens=None):
        # motion:      [B, T, 263]
        # clip_embeds: [B, N, clip_dim]  (CLIP-encoded segment texts)
        # seg_mask:    [B, N] bool
        # m_lens:      [B]

        m_lens_down = m_lens // (2 ** self.down_t) if m_lens is not None else None

        # 1. Encode
        feat = self.encode(motion, m_lens)        # [B, T/4, latent_dim]

        # 2. Attention pooling → N tokens
        tokens = self.seg_pool(clip_embeds, feat, seg_mask)  # [B, N, latent_dim]

        # 3. VQ quantization
        tokens_q, commit_loss, _, vq_perplexity, vq_n_active = self.vq(tokens, seg_mask)

        # 4. scale0 upsample: N → T/4  →  f̂0
        T_down = feat.shape[1]
        f_hat_0 = F.interpolate(
            tokens_q.permute(0, 2, 1),   # [B, latent_dim, N]
            size=T_down, mode='linear', align_corners=False
        )                                # [B, latent_dim, T/4]

        # mask padding frames
        if m_lens_down is not None:
            mask = torch.arange(T_down, device=feat.device).unsqueeze(0) < m_lens_down.unsqueeze(1)
            f_hat_0 = f_hat_0 * mask.unsqueeze(1).float()

        # 5. residual = f - f̂0  →  scale1~4 (MSQuantizer)
        # detach f_hat_0: commit_hrv gradient should not flow back through seg_pool/vq
        feat_perm = feat.permute(0, 2, 1)          # [B, latent_dim, T/4]
        residual  = feat_perm - f_hat_0.detach()

        x_quantized, commit_loss_hrv, _ = self.quantizer(
            residual, temperature=0.5, m_lens=m_lens_down,
            start_drop=0, quantize_dropout_prob=0.0
        )                                           # [B, latent_dim, T/4]

        # 6. 전체 reconstruction feature = f̂0 + f̂1~4
        f_hat_total = f_hat_0 + x_quantized        # [B, latent_dim, T/4]

        # 7. Decode
        x_recon = self.decoder(f_hat_total, m_lens_down)  # [B, T, 263]

        # 6. Losses
        # reconstruction
        if m_lens is not None:
            t_mask = torch.arange(motion.shape[1], device=motion.device).unsqueeze(0) < m_lens.unsqueeze(1)
            l_recon = F.smooth_l1_loss(x_recon[t_mask], motion.float()[t_mask])
        else:
            l_recon = F.smooth_l1_loss(x_recon, motion.float())

        # alignment loss: encoder feature를 N등분 avg pool → triplet
        # SimpleVQ/SegmentAttnPool과 독립 → codebook collapse와 무관하게 학습
        seg_feats = self._temporal_seg_pool(feat, seg_mask, m_lens_down)  # [B, N, D]
        l_align   = self._seg_align_loss(seg_feats, clip_embeds, seg_mask)

        # clamp commit_hrv: torch.clamp saturates gradient to 0 when value > max
        # → bad batches (commit_hrv spike) don't destabilize encoder
        commit_loss_hrv_stable = commit_loss_hrv.clamp(max=5.0)

        loss = (l_recon
                + self.lambda_commit     * commit_loss
                + self.lambda_commit_hrv * commit_loss_hrv_stable
                + self.lambda_align      * l_align)

        return x_recon, loss, {
            'l_recon':      l_recon.item(),
            'l_commit_seg': commit_loss.item(),
            'l_commit_hrv': commit_loss_hrv.item(),
            'l_align':      l_align.item(),
            'vq_perplexity': vq_perplexity.item(),
            'vq_n_active':   vq_n_active.item(),
        }
