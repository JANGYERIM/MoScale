"""
Plug-in Alignment Training
- HRVQVAE encoder 완전 frozen → m_i 고정
- align_proj_motion + align_proj_text + CLIP last 2 layer만 학습
- Loss: InfoNCE(vm, vt)
- 저장: checkpoint_dir/humanml3d/seg_align_plugin/{ckpt_name}.tar
"""
import os
import json
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from model.vq.hrvqvae import HRVQVAE
from dataset.seg_dataset import SegMotionDataset, seg_collate_fn
from config.load_config import load_config
from utils.get_opt import get_opt
from utils.fixseeds import fixseed
import wandb


# ── helpers ────────────────────────────────────────────────────────────────────

def compute_clip_mean(tokenizer, clip_model, jsonl_path, device, max_samples=5000):
    texts = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            for cap in d.get('captions', []):
                for seg in cap.get('segments', []):
                    texts.append(seg)
            if len(texts) >= max_samples:
                break
    texts = texts[:max_samples]
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            tok = tokenizer(texts[i:i+256], padding=True, truncation=True,
                            max_length=77, return_tensors='pt').to(device)
            all_embeds.append(clip_model(**tok).pooler_output.cpu())
    mean_embed = torch.cat(all_embeds, dim=0).mean(0)
    print(f"[CLIP centering] computed mean over {len(texts)} texts")
    return mean_embed.to(device)


def encode_segments(tokenizer, clip_model, seg_texts_batch, device, clip_mean=None):
    """gradient flows through CLIP last 2 layers"""
    B, max_n = len(seg_texts_batch), len(seg_texts_batch[0])
    flat_texts = [t for sample in seg_texts_batch for t in sample]
    tok = tokenizer(flat_texts, padding=True, truncation=True,
                    max_length=77, return_tensors='pt').to(device)
    embeds = clip_model(**tok).pooler_output
    if clip_mean is not None:
        embeds = F.normalize(embeds - clip_mean.unsqueeze(0), dim=-1)
    return embeds.reshape(B, max_n, -1)


def temporal_seg_pool(feat, seg_mask, m_lens_down):
    """encoder feat [B, T, D] → segment avg pool [B, N, D]"""
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
    return out


def infonce_loss(vm, vt, temperature=0.07):
    """vm, vt: [V, A] L2-normalized"""
    V = vm.shape[0]
    if V < 2:
        return vm.sum() * 0.
    sim    = vm @ vt.T / temperature
    labels = torch.arange(V, device=vm.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) * 0.5


def log_cosim(vm, vt, epoch, global_step, wandb_run):
    V = vm.shape[0]
    cos_sims = (vm * vt).sum(-1).cpu()
    print(f"\n{'='*60}")
    print(f"[CosSim Ep {epoch:03d}] valid_pairs={V}")
    print(f"  (m_i,t_i) diag : mean={cos_sims.mean():.4f} | std={cos_sims.std():.4f} | "
          f"min={cos_sims.min():.4f} | max={cos_sims.max():.4f}")
    show_n = min(V, 32)
    row = "  "
    for i in range(show_n):
        row += f"[{i:3d}]{cos_sims[i]:+.3f} "
        if (i + 1) % 8 == 0:
            print(row); row = "  "
    if row.strip():
        print(row)
    if V > 32:
        print(f"  ... ({V - 32} more pairs)")

    mask_off = ~torch.eye(V, dtype=torch.bool)
    sim_tt = (vt @ vt.T)[mask_off]
    sim_mm = (vm @ vm.T)[mask_off]
    print(f"  (t_i,t_j) off-diag: mean={sim_tt.mean():.4f} | std={sim_tt.std():.4f} | "
          f"min={sim_tt.min():.4f} | max={sim_tt.max():.4f}")
    print(f"  (m_i,m_j) off-diag: mean={sim_mm.mean():.4f} | std={sim_mm.std():.4f} | "
          f"min={sim_mm.min():.4f} | max={sim_mm.max():.4f}")
    print('='*60 + '\n')

    wandb_run.log({
        'plugin/cosim_pos_mean':  cos_sims.mean().item(),
        'plugin/cosim_pos_std':   cos_sims.std().item(),
        'plugin/tt_offdiag_mean': sim_tt.mean().item(),
        'plugin/mm_offdiag_mean': sim_mm.mean().item(),
    }, step=global_step)


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # config
    seg_cfg = load_config('config/train_segvqvae.yaml')
    hrv_cfg = load_config('checkpoint_dir/humanml3d/hrvqvae/T_VQ_V1/train_hrvqvae.yaml')

    ckpt_name = seg_cfg.training.get('plugin_ckpt_name', 'align_plugin_v1')
    ckpt_dir  = pjoin(seg_cfg.exp.root_ckpt_dir, seg_cfg.data.name, 'seg_align_plugin')
    os.makedirs(ckpt_dir, exist_ok=True)

    fixseed(seg_cfg.exp.seed)
    device = torch.device(seg_cfg.exp.device)

    run = wandb.init(project='SegAlign_Plugin', dir=ckpt_dir,
                     config=dict(seg_cfg), name=ckpt_name)

    # dataset
    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device, data_root=seg_cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std  = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))
    wrapper_opt.motion_dir = pjoin(seg_cfg.data.root_dir, 'new_joint_vecs')
    wrapper_opt.text_dir   = pjoin(seg_cfg.data.root_dir, 'texts')

    train_dataset = SegMotionDataset(
        wrapper_opt, mean, std,
        split_file=pjoin(seg_cfg.data.root_dir, 'train.txt'),
        seg_jsonl_path=seg_cfg.data.seg_jsonl_train,
        max_n_seg=seg_cfg.data.max_n_seg,
    )
    train_loader = DataLoader(train_dataset, batch_size=seg_cfg.training.batch_size,
                              shuffle=True, num_workers=8, drop_last=True,
                              collate_fn=seg_collate_fn)

    # ── HRVQVAE encoder (frozen) ──────────────────────────────────────────────
    hrv = HRVQVAE(hrv_cfg,
                  input_width=263,
                  down_t=hrv_cfg.model.down_t,
                  stride_t=hrv_cfg.model.stride_t,
                  width=hrv_cfg.model.width,
                  depth=hrv_cfg.model.depth,
                  dilation_growth_rate=hrv_cfg.model.dilation_growth_rate,
                  activation=hrv_cfg.model.vq_act,
                  use_attn=hrv_cfg.model.use_attn).to(device)

    hrv_ckpt_path = 'checkpoint_dir/humanml3d/hrvqvae/T_VQ_V1/model/net_best_fid.tar'
    hrv_ckpt = torch.load(hrv_ckpt_path, map_location=device)
    hrv.load_state_dict(hrv_ckpt['vq_model'])
    hrv.eval()
    for p in hrv.parameters():
        p.requires_grad = False
    print(f"[HRVQVAE] loaded & frozen: {hrv_ckpt_path}")
    down_t = hrv_cfg.model.down_t   # 2 → 4x downsample

    # ── align projectors (standalone) ────────────────────────────────────────
    align_dim = seg_cfg.model.get('align_dim', 256)
    latent_dim = hrv_cfg.quantizer.code_dim  # 512
    clip_dim   = seg_cfg.model.clip_dim      # 512

    align_proj_motion = nn.Linear(latent_dim, align_dim).to(device)
    align_proj_text   = nn.Linear(clip_dim,   align_dim).to(device)
    print(f"align_proj_motion: {latent_dim}→{align_dim}  |  align_proj_text: {clip_dim}→{align_dim}")

    # ── CLIP (last 2 layer fine-tune) ─────────────────────────────────────────
    clip_tokenizer  = CLIPTokenizer.from_pretrained(seg_cfg.model.clip_model)
    clip_text_model = CLIPTextModel.from_pretrained(seg_cfg.model.clip_model).to(device)
    for p in clip_text_model.parameters():
        p.requires_grad = False
    for layer in clip_text_model.text_model.encoder.layers[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in clip_text_model.text_model.final_layer_norm.parameters():
        p.requires_grad = True
    clip_trainable = [p for p in clip_text_model.parameters() if p.requires_grad]
    print(f"[CLIP] fine-tune last 2 layers: "
          f"{sum(p.numel() for p in clip_trainable)/1e6:.2f}M params")

    # clip mean (centering)
    clip_mean_path = 'config/clip_mean.pt'
    clip_text_model.eval()
    if os.path.exists(clip_mean_path):
        clip_mean = torch.load(clip_mean_path, map_location=device)
        print(f"[CLIP centering] loaded from {clip_mean_path}")
    else:
        clip_mean = compute_clip_mean(clip_tokenizer, clip_text_model,
                                      seg_cfg.data.seg_jsonl_train, device)
        torch.save(clip_mean, clip_mean_path)

    # ── optimizer ─────────────────────────────────────────────────────────────
    lr      = seg_cfg.training.lr
    lr_clip = seg_cfg.training.get('lr_clip', 1e-5)
    optimizer = torch.optim.AdamW([
        {'params': list(align_proj_motion.parameters()) +
                   list(align_proj_text.parameters()),  'lr': lr},
        {'params': clip_trainable,                       'lr': lr_clip},
    ], weight_decay=seg_cfg.training.weight_decay)

    num_epochs  = seg_cfg.training.get('plugin_epochs', seg_cfg.training.get('stage1_epochs', 100))
    global_step = 0

    for epoch in range(num_epochs):
        align_proj_motion.train()
        align_proj_text.train()
        clip_text_model.train()
        ep_loss, ep_steps = 0., 0
        log_cosim_this_epoch = (epoch % 20 == 0)

        for batch_idx, (motion, seg_texts, m_lens, seg_mask, n_valid) in enumerate(train_loader):
            motion   = motion.to(device)
            seg_mask = seg_mask.to(device)
            m_lens   = m_lens.to(device).long()
            m_lens_down = m_lens // (2 ** down_t)

            # m_i: HRVQVAE encoder (frozen, no_grad)
            with torch.no_grad():
                x_in = motion.permute(0, 2, 1).float()          # [B, 263, T]
                feat = hrv.encoder(x_in, m_lens)                 # [B, 512, T/4]
                feat = feat.permute(0, 2, 1)                     # [B, T/4, 512]
                seg_feats = temporal_seg_pool(feat, seg_mask, m_lens_down)  # [B, N, 512]

            # t_i: CLIP (gradient through last 2 layers)
            seg_texts_list = [list(col) for col in zip(*seg_texts)]
            clip_embeds = encode_segments(clip_tokenizer, clip_text_model,
                                          seg_texts_list, device, clip_mean)  # [B, N, 512]

            # projection → shared space
            B, N, _ = seg_feats.shape
            flat_mask = seg_mask.reshape(B * N).bool()

            vm = F.normalize(align_proj_motion(seg_feats).reshape(B*N, -1)[flat_mask], dim=-1)
            vt = F.normalize(align_proj_text(clip_embeds).reshape(B*N, -1)[flat_mask], dim=-1)

            # cosim logging (20에폭 주기, 첫 배치)
            if log_cosim_this_epoch and batch_idx == 0:
                with torch.no_grad():
                    log_cosim(vm.detach(), vt.detach(), epoch, global_step, run)
                log_cosim_this_epoch = False

            loss = infonce_loss(vm, vt, temperature=0.07)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(align_proj_motion.parameters()) +
                list(align_proj_text.parameters()) + clip_trainable, 1.0)
            optimizer.step()

            ep_loss  += loss.item()
            ep_steps += 1
            print(f"[Plugin] Ep {epoch:03d} | step {global_step} | align {loss.item():.4f}")

            if global_step % seg_cfg.training.log_every == 0:
                run.log({'plugin/align': loss.item(), 'epoch': epoch}, step=global_step)
            global_step += 1

        avg = ep_loss / ep_steps
        print(f"\n[Plugin Ep {epoch:03d}] avg align {avg:.4f}\n")
        run.log({'plugin/epoch_align': avg, 'epoch': epoch}, step=global_step)

    # ── save ──────────────────────────────────────────────────────────────────
    save_path = pjoin(ckpt_dir, f'{ckpt_name}.tar')
    torch.save({
        'epoch':             num_epochs,
        'align_proj_motion': align_proj_motion.state_dict(),
        'align_proj_text':   align_proj_text.state_dict(),
        'clip_last2_layers': [
            clip_text_model.text_model.encoder.layers[-2].state_dict(),
            clip_text_model.text_model.encoder.layers[-1].state_dict(),
        ],
        'clip_final_layer_norm': clip_text_model.text_model.final_layer_norm.state_dict(),
        'hrv_ckpt_path':     hrv_ckpt_path,
        'align_dim':         align_dim,
    }, save_path)
    print(f"\n[Plugin 완료] saved → {save_path}")
    wandb.finish()
