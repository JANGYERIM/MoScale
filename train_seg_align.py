"""
Stage 1: Encoder + alignment projector 학습 (triplet loss only)
- SegmentAttnPool / SimpleVQ / MSQuantizer / Decoder는 학습하지 않음
- 완료 후 encoder checkpoint → config/stage1_encoder.tar
"""
import os
import json
import shutil
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from model.vq.seg_vqvae import SegVQVAE
from dataset.seg_dataset import SegMotionDataset, seg_collate_fn
from config.load_config import load_config
from utils.get_opt import get_opt
from utils.fixseeds import fixseed
import wandb


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
    B, max_n = len(seg_texts_batch), len(seg_texts_batch[0])
    flat_texts = [t for sample in seg_texts_batch for t in sample]
    # no_grad 제거: CLIP last 2 layer gradient 흘려야 함
    tok = tokenizer(flat_texts, padding=True, truncation=True,
                    max_length=77, return_tensors='pt').to(device)
    embeds = clip_model(**tok).pooler_output
    if clip_mean is not None:
        embeds = F.normalize(embeds - clip_mean.unsqueeze(0), dim=-1)
    return embeds.reshape(B, max_n, -1)


if __name__ == '__main__':
    cfg = load_config('config/train_segvqvae.yaml')
    ckpt_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'segvqvae', cfg.exp.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    fixseed(cfg.exp.seed)
    device = torch.device(cfg.exp.device)

    wandb.init(project='SegVQVAE_align', dir=ckpt_dir,
               config=dict(cfg), name=cfg.exp.name + '_stage1')

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device, data_root=cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std  = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))
    wrapper_opt.motion_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')
    wrapper_opt.text_dir   = pjoin(cfg.data.root_dir, 'texts')

    train_dataset = SegMotionDataset(
        wrapper_opt, mean, std,
        split_file=pjoin(cfg.data.root_dir, 'train.txt'),
        seg_jsonl_path=cfg.data.seg_jsonl_train,
        max_n_seg=cfg.data.max_n_seg,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=8, drop_last=True,
                              collate_fn=seg_collate_fn)

    # CLIP: 마지막 2 레이어 + final_layer_norm만 fine-tune
    clip_tokenizer  = CLIPTokenizer.from_pretrained(cfg.model.clip_model)
    clip_text_model = CLIPTextModel.from_pretrained(cfg.model.clip_model).to(device)
    for p in clip_text_model.parameters():
        p.requires_grad = False
    for layer in clip_text_model.text_model.encoder.layers[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in clip_text_model.text_model.final_layer_norm.parameters():
        p.requires_grad = True
    clip_trainable = [p for p in clip_text_model.parameters() if p.requires_grad]
    print(f"[CLIP] fine-tune last 2 layers: "
          f"{sum(p.numel() for p in clip_trainable)/1e6:.2f}M params trainable")

    clip_mean_path = 'config/clip_mean.pt'
    clip_text_model.eval()
    if os.path.exists(clip_mean_path):
        clip_mean = torch.load(clip_mean_path, map_location=device)
        print(f"[CLIP centering] loaded from {clip_mean_path}")
    else:
        clip_mean = compute_clip_mean(clip_tokenizer, clip_text_model,
                                      cfg.data.seg_jsonl_train, device)
        torch.save(clip_mean, clip_mean_path)

    # 모델 생성 (전체 구조 유지, encoder + align projectors만 학습)
    net = SegVQVAE(cfg).to(device)
    print(f"SegVQVAE params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    # Stage 1: encoder + align_proj_motion/text 만 학습
    # seg_pool / vq / quantizer / decoder 는 freeze
    for name, p in net.named_parameters():
        if any(name.startswith(k) for k in
               ['encoder', 'align_proj_motion', 'align_proj_text']):
            p.requires_grad = True
        else:
            p.requires_grad = False

    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    print(f"[Stage 1] trainable modules: {set(n.split('.')[0] for n in trainable)}")

    num_epochs = cfg.training.get('stage1_epochs', 50)
    lr_clip    = cfg.training.get('lr_clip', 1e-5)
    optimizer  = torch.optim.AdamW([
        {'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': cfg.training.lr},
        {'params': clip_trainable, 'lr': lr_clip},
    ], weight_decay=cfg.training.weight_decay)

    global_step = 0
    for epoch in range(num_epochs):
        net.train()
        clip_text_model.train()   # last 2 layers 업데이트 활성화
        ep_align, ep_steps = 0., 0
        log_cosim_this_epoch = (epoch % 20 == 0)

        for batch_idx, (motion, seg_texts, m_lens, seg_mask, n_valid) in enumerate(train_loader):
            motion   = motion.to(device)
            seg_mask = seg_mask.to(device)
            m_lens   = m_lens.to(device)

            seg_texts_list = [list(col) for col in zip(*seg_texts)]
            clip_embeds = encode_segments(clip_tokenizer, clip_text_model,
                                          seg_texts_list, device, clip_mean)

            # encoder forward (gradient 흐름)
            feat = net.encode(motion, m_lens)   # [B, T/4, D]

            # temporal seg pool
            m_lens_down = m_lens // (2 ** net.down_t)
            seg_feats = net._temporal_seg_pool(feat, seg_mask, m_lens_down)

            # 20에폭 주기, 첫 배치에서 각 (m_i, t_i) cosine similarity 출력
            if log_cosim_this_epoch and batch_idx == 0:
                with torch.no_grad():
                    B, N, _ = seg_feats.shape
                    m_proj_log = F.normalize(net.align_proj_motion(seg_feats.detach()), dim=-1)
                    t_proj_log = F.normalize(net.align_proj_text(clip_embeds.detach()), dim=-1)
                    flat_mask = (seg_mask.reshape(B * N).bool() if seg_mask is not None
                                 else torch.ones(B * N, dtype=torch.bool, device=seg_feats.device))
                    vm = m_proj_log.reshape(B * N, -1)[flat_mask]   # [V, A]
                    vt = t_proj_log.reshape(B * N, -1)[flat_mask]   # [V, A]
                    cos_sims = (vm * vt).sum(-1).cpu()               # [V]
                    V = len(cos_sims)
                    # (m_i, t_i) 양성쌍 cosine similarity
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

                    # t_i 간 pairwise similarity (off-diagonal): centering 효과 확인
                    sim_tt   = vt @ vt.T                          # [V, V]
                    mask_off = ~torch.eye(V, dtype=torch.bool)
                    tt_off   = sim_tt[mask_off]
                    print(f"  (t_i,t_j) off-diag: mean={tt_off.mean():.4f} | std={tt_off.std():.4f} | "
                          f"min={tt_off.min():.4f} | max={tt_off.max():.4f}")

                    # m_i 간 pairwise similarity (off-diagonal): motion 다양성 확인
                    sim_mm = vm @ vm.T
                    mm_off = sim_mm[mask_off]
                    print(f"  (m_i,m_j) off-diag: mean={mm_off.mean():.4f} | std={mm_off.std():.4f} | "
                          f"min={mm_off.min():.4f} | max={mm_off.max():.4f}")
                    print('='*60 + '\n')

                    wandb.log({
                        's1/cosim_pos_mean':  cos_sims.mean().item(),
                        's1/cosim_pos_std':   cos_sims.std().item(),
                        's1/cosim_pos_min':   cos_sims.min().item(),
                        's1/cosim_pos_max':   cos_sims.max().item(),
                        's1/tt_offdiag_mean': tt_off.mean().item(),
                        's1/tt_offdiag_std':  tt_off.std().item(),
                        's1/tt_offdiag_min':  tt_off.min().item(),
                        's1/mm_offdiag_mean': mm_off.mean().item(),
                        's1/mm_offdiag_std':  mm_off.std().item(),
                    }, step=global_step)
                log_cosim_this_epoch = False  # 해당 에폭 첫 배치만 출력

            # triplet alignment loss
            loss = net._seg_align_loss(seg_feats, clip_embeds, seg_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(net.parameters()) + clip_trainable, 1.0)
            optimizer.step()

            ep_align += loss.item()
            ep_steps += 1
            print(f"[S1] Ep {epoch:03d} | step {global_step} | align {loss.item():.4f}")

            if global_step % cfg.training.log_every == 0:
                wandb.log({'s1/align': loss.item(), 'epoch': epoch}, step=global_step)
            global_step += 1

        avg = ep_align / ep_steps
        print(f"\n[Stage1 Ep {epoch:03d}] align {avg:.4f}\n")
        wandb.log({'s1/epoch_align': avg, 'epoch': epoch}, step=global_step)

    # Stage 1 encoder + projectors 저장
    ckpt_name = cfg.training.get('stage1_ckpt_name', 'stage1_encoder')
    save_path  = pjoin(ckpt_dir, f'{ckpt_name}.tar')
    torch.save({
        'epoch': num_epochs,
        'encoder':              net.encoder.state_dict(),
        'align_proj_motion':    net.align_proj_motion.state_dict(),
        'align_proj_text':      net.align_proj_text.state_dict(),
        # fine-tuned CLIP layers (test time에도 동일하게 써야 함)
        'clip_last2_layers': [
            clip_text_model.text_model.encoder.layers[-2].state_dict(),
            clip_text_model.text_model.encoder.layers[-1].state_dict(),
        ],
        'clip_final_layer_norm': clip_text_model.text_model.final_layer_norm.state_dict(),
    }, save_path)
    print(f"\n[Stage 1 완료] saved → {save_path}")
