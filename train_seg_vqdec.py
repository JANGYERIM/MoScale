"""
Stage 2: MSQuantizer + Decoder 학습
- Encoder: Stage 1 checkpoint 로드 후 freeze
- SegmentAttnPool / SimpleVQ 제거
- scale0 = temporal_split(feat) avg pool → upsample (VQ 없음)
- scale1~4 = MSQuantizer(residual)
- Loss: l_recon + l_commit_hrv
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

from model.vq.seg_vqvae import SegVQVAE
from dataset.seg_dataset import SegMotionDataset, seg_collate_fn
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from config.load_config import load_config
from utils.get_opt import get_opt
from utils.fixseeds import fixseed
from utils.metrics import (calculate_frechet_distance, calculate_R_precision_gpu,
                            euclidean_distance_matrix_gpu, calculate_activation_statistics_gpu,
                            calculate_diversity_gpu)
import wandb


@torch.no_grad()
def evaluate_seg_vqdec(net, eval_loader, eval_wrapper, ep, global_step, best_fid, device):
    net.eval()
    motion_annotation_list, motion_pred_list = [], []
    R_prec_real = torch.zeros(3, device=device)
    R_prec_pred = torch.zeros(3, device=device)
    match_real = match_pred = nb_sample = 0.

    dataset  = eval_loader.dataset
    std_gpu  = torch.from_numpy(dataset.std).float().to(device)
    mean_gpu = torch.from_numpy(dataset.mean).float().to(device)

    for batch in eval_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = batch
        motion   = motion.to(device)
        m_length = m_length.to(device)
        B = motion.shape[0]

        # SegVQVAE forward: temporal_seg_pool으로 coarse → residual 양자화 → recon
        m_lens_down = m_length // (2 ** net.down_t)
        feat      = net.encode(motion, m_length)
        T_down    = feat.shape[1]

        # N=1 segment (full motion을 1개 segment로 취급)
        seg_mask  = torch.ones(B, 1, dtype=torch.bool, device=device)
        seg_feats = net._temporal_seg_pool(feat, seg_mask, m_lens_down)

        f_hat_0 = F.interpolate(
            seg_feats.permute(0, 2, 1), size=T_down, mode='linear', align_corners=False
        )
        pad_mask = torch.arange(T_down, device=device).unsqueeze(0) < m_lens_down.unsqueeze(1)
        f_hat_0  = f_hat_0 * pad_mask.unsqueeze(1).float()

        residual    = feat.permute(0, 2, 1) - f_hat_0
        x_quantized, _, _ = net.quantizer(residual, temperature=0., m_lens=m_lens_down,
                                           start_drop=0, quantize_dropout_prob=0.0)
        x_recon = net.decoder(f_hat_0 + x_quantized, m_lens_down)

        et,      em      = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion,  m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, x_recon, m_length)

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        R_prec_real += calculate_R_precision_gpu(et,      em,      top_k=3, sum_all=True)
        R_prec_pred += calculate_R_precision_gpu(et_pred, em_pred, top_k=3, sum_all=True)
        match_real  += euclidean_distance_matrix_gpu(et,      em     ).trace().item()
        match_pred  += euclidean_distance_matrix_gpu(et_pred, em_pred).trace().item()
        nb_sample   += B

    em_all   = torch.cat(motion_annotation_list, dim=0)
    pred_all = torch.cat(motion_pred_list,       dim=0)

    gt_mu, gt_cov = calculate_activation_statistics_gpu(em_all)
    mu,    cov    = calculate_activation_statistics_gpu(pred_all)
    fid           = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    div_real      = calculate_diversity_gpu(em_all,   300 if nb_sample > 300 else 100)
    div           = calculate_diversity_gpu(pred_all, 300 if nb_sample > 300 else 100)

    top123_real = (R_prec_real / nb_sample).cpu().numpy()
    top123      = (R_prec_pred / nb_sample).cpu().numpy()
    match_real /= nb_sample
    match_pred /= nb_sample

    print(f"\n[Eval Ep {ep:03d}] FID {fid:.4f} | "
          f"Top1 {top123[0]:.4f} Top2 {top123[1]:.4f} Top3 {top123[2]:.4f} | "
          f"Div {div:.4f} | Match {match_pred:.4f}\n")

    wandb.log({'eval/FID': fid, 'eval/Top1': top123[0], 'eval/Top2': top123[1],
               'eval/Top3': top123[2], 'eval/Div': div, 'eval/Match': match_pred,
               'epoch': ep}, step=global_step)

    if fid < best_fid:
        print(f"  --> FID improved: {best_fid:.4f} → {fid:.4f}")
        best_fid = fid

    return fid, top123[0], best_fid


if __name__ == '__main__':
    cfg = load_config('config/train_segvqvae.yaml')
    ckpt_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'segvqvae', cfg.exp.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    fixseed(cfg.exp.seed)
    device = torch.device(cfg.exp.device)

    wandb.init(project='SegVQVAE_vqdec', dir=ckpt_dir,
               config=dict(cfg), name=cfg.exp.name + '_stage2')

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device, data_root=cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std  = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))
    wrapper_opt.motion_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')
    wrapper_opt.text_dir   = pjoin(cfg.data.root_dir, 'texts')

    from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test',
                                               device=device, data_root=cfg.data.root_dir)

    train_dataset = SegMotionDataset(
        wrapper_opt, mean, std,
        split_file=pjoin(cfg.data.root_dir, 'train.txt'),
        seg_jsonl_path=cfg.data.seg_jsonl_train,
        max_n_seg=cfg.data.max_n_seg,
    )
    val_dataset = SegMotionDataset(
        wrapper_opt, mean, std,
        split_file=pjoin(cfg.data.root_dir, 'val.txt'),
        seg_jsonl_path=cfg.data.seg_jsonl_val,
        max_n_seg=cfg.data.max_n_seg,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, num_workers=8, drop_last=True,
                              collate_fn=seg_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=4, drop_last=False,
                              collate_fn=seg_collate_fn)

    net = SegVQVAE(cfg).to(device)
    print(f"SegVQVAE params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    s2_name = cfg.training.get('stage2_ckpt_name', 'stage2_vqdec')

    # HRVQVAE plug-in: encoder / codebook / decoder 모두 로드 후 freeze
    hrv_ckpt_path = cfg.training.get('hrv_ckpt_path', None)
    if hrv_ckpt_path:
        hrv_ckpt = torch.load(hrv_ckpt_path, map_location=device)
        hrv_model = hrv_ckpt['vq_model']

        encoder_state = {k[len('encoder.'):]: v
                         for k, v in hrv_model.items() if k.startswith('encoder.')}
        net.encoder.load_state_dict(encoder_state)
        print(f"[Stage 2] loaded HRVQVAE encoder → frozen")

        # codebook은 residual(encoder_output - f_hat_0) 분포로 새로 학습
        # net.quantizer.codebook.copy_(hrv_model['quantizer.codebook'])
        # net.quantizer.init = True
        # net.quantizer.codebook_frozen = True
        # print(f"[Stage 2] loaded HRVQVAE codebook → frozen")

        decoder_state = {k[len('decoder.'):]: v
                         for k, v in hrv_model.items() if k.startswith('decoder.')}
        net.decoder.load_state_dict(decoder_state)
        print(f"[Stage 2] loaded HRVQVAE decoder from {hrv_ckpt_path} → frozen")
    else:
        print("[Stage 2] hrv_ckpt_path not set → all components train from scratch")

    # plugin align_proj_text 로드
    plugin_ckpt_path = cfg.training.get('plugin_ckpt_path', None)
    if plugin_ckpt_path:
        plugin_ckpt = torch.load(plugin_ckpt_path, map_location=device)
        net.align_proj_text.load_state_dict(plugin_ckpt['align_proj_text'])
        print(f"[Stage 2] loaded plugin align_proj_text from {plugin_ckpt_path}")

    # freeze: encoder, align projectors, seg_pool, vq, decoder, quantizer.quant_resi
    # codebook만 EMA로 residual 분포에 수렴 (gradient 충돌 없음)
    for name, p in net.named_parameters():
        if any(name.startswith(k) for k in
               ['encoder', 'align_proj_motion', 'align_proj_text',
                'seg_pool', 'vq', 'decoder', 'quantizer']):
            p.requires_grad = False

    trainable = {n.split('.')[0] for n, p in net.named_parameters() if p.requires_grad}
    print(f"[Stage 2] trainable: {trainable}")

    # quant_resi frozen이므로 optimizer 불필요 (codebook은 EMA buffer로 자동 업데이트)
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    # )

    best_val_loss = float('inf')
    best_fid      = float('inf')
    global_step   = 0

    for epoch in range(cfg.training.num_epochs):
        net.train()
        net.encoder.eval()   # encoder는 항상 eval

        ep_recon = ep_commit_hrv = ep_total = ep_perplexity = 0.
        ep_steps = 0

        for motion, seg_texts, m_lens, seg_mask, n_valid in train_loader:
            motion   = motion.to(device)
            seg_mask = seg_mask.to(device)
            m_lens   = m_lens.to(device)
            m_lens_down = m_lens // (2 ** net.down_t)

            # encoder: no_grad (frozen)
            with torch.no_grad():
                feat = net.encode(motion, m_lens)   # [B, T/4, D]

            T_down = feat.shape[1]

            # scale0: temporal split avg pool → upsample
            # SegmentAttnPool / SimpleVQ 없음 → codebook collapse 없음
            with torch.no_grad():
                seg_feats = net._temporal_seg_pool(feat, seg_mask, m_lens_down)  # [B, N, D]

            f_hat_0 = F.interpolate(
                seg_feats.permute(0, 2, 1),   # [B, D, N]
                size=T_down, mode='linear', align_corners=False
            )                                  # [B, D, T/4]

            if m_lens_down is not None:
                pad_mask = (torch.arange(T_down, device=device).unsqueeze(0)
                            < m_lens_down.unsqueeze(1))
                f_hat_0 = f_hat_0 * pad_mask.unsqueeze(1).float()

            # scale1~4: MSQuantizer on residual
            residual = feat.permute(0, 2, 1) - f_hat_0.detach()
            x_quantized, commit_loss_hrv, perplexity = net.quantizer(
                residual, temperature=0.5, m_lens=m_lens_down,
                start_drop=0, quantize_dropout_prob=0.0
            )

            f_hat_total = f_hat_0 + x_quantized
            x_recon = net.decoder(f_hat_total, m_lens_down)

            if m_lens is not None:
                t_mask = (torch.arange(motion.shape[1], device=device).unsqueeze(0)
                          < m_lens.unsqueeze(1))
                l_recon = F.smooth_l1_loss(x_recon[t_mask], motion.float()[t_mask])
            else:
                l_recon = F.smooth_l1_loss(x_recon, motion.float())

            commit_loss_hrv_stable = commit_loss_hrv.clamp(max=5.0)
            loss = l_recon + net.lambda_commit_hrv * commit_loss_hrv_stable

            # quant_resi frozen → trainable params 없음, codebook은 EMA로 업데이트됨
            # backward/optimizer step 불필요 (forward에서 EMA 자동 반영)

            ep_recon        += l_recon.item()
            ep_commit_hrv   += commit_loss_hrv.item()
            ep_total        += loss.item()
            ep_perplexity   += perplexity.item()
            ep_steps        += 1

            print(f"[S2] Ep {epoch:03d} | step {global_step} | "
                  f"recon {l_recon.item():.4f} | "
                  f"commit_hrv {commit_loss_hrv.item():.4f} | "
                  f"ppl {perplexity.item():.1f}")

            if global_step % cfg.training.log_every == 0:
                wandb.log({
                    's2/recon':       l_recon.item(),
                    's2/commit_hrv':  commit_loss_hrv.item(),
                    's2/perplexity':  perplexity.item(),
                    'epoch': epoch,
                }, step=global_step)
            global_step += 1

        print(f"\n[Stage2 Ep {epoch:03d}] "
              f"total {ep_total/ep_steps:.4f} | "
              f"recon {ep_recon/ep_steps:.4f} | "
              f"commit_hrv {ep_commit_hrv/ep_steps:.4f} | "
              f"ppl {ep_perplexity/ep_steps:.1f}\n")
        wandb.log({
            's2/epoch_total':      ep_total      / ep_steps,
            's2/epoch_recon':      ep_recon      / ep_steps,
            's2/epoch_commit_hrv': ep_commit_hrv / ep_steps,
            's2/epoch_perplexity': ep_perplexity / ep_steps,
            'epoch': epoch,
        }, step=global_step)

        # validation
        if (epoch + 1) % cfg.training.val_every == 0:
            net.eval()
            val_recon = 0.
            n_val = 0
            with torch.no_grad():
                for motion, seg_texts, m_lens, seg_mask, n_valid in val_loader:
                    motion      = motion.to(device)
                    seg_mask    = seg_mask.to(device)
                    m_lens      = m_lens.to(device)
                    m_lens_down = m_lens // (2 ** net.down_t)

                    feat      = net.encode(motion, m_lens)
                    T_down    = feat.shape[1]
                    seg_feats = net._temporal_seg_pool(feat, seg_mask, m_lens_down)

                    f_hat_0 = F.interpolate(
                        seg_feats.permute(0, 2, 1),
                        size=T_down, mode='linear', align_corners=False
                    )
                    if m_lens_down is not None:
                        pad_mask = (torch.arange(T_down, device=device).unsqueeze(0)
                                    < m_lens_down.unsqueeze(1))
                        f_hat_0 = f_hat_0 * pad_mask.unsqueeze(1).float()

                    residual    = feat.permute(0, 2, 1) - f_hat_0
                    x_quantized, _, _ = net.quantizer(
                        residual, temperature=0.5, m_lens=m_lens_down,
                        start_drop=0, quantize_dropout_prob=0.0
                    )
                    x_recon = net.decoder(f_hat_0 + x_quantized, m_lens_down)

                    if m_lens is not None:
                        t_mask = (torch.arange(motion.shape[1], device=device).unsqueeze(0)
                                  < m_lens.unsqueeze(1))
                        l_recon = F.smooth_l1_loss(x_recon[t_mask], motion.float()[t_mask])
                    else:
                        l_recon = F.smooth_l1_loss(x_recon, motion.float())

                    val_recon += l_recon.item()
                    n_val     += 1

            val_recon /= n_val
            print(f"[Val S2 Ep {epoch:03d}] recon {val_recon:.4f}")
            wandb.log({'s2/val_recon': val_recon, 'epoch': epoch}, step=global_step)

            if val_recon < best_val_loss:
                best_val_loss = val_recon
                torch.save({'epoch': epoch, 'model': net.state_dict()},
                           pjoin(ckpt_dir, f'{s2_name}_best.tar'))
                print(f"  --> best saved (recon={val_recon:.4f})")

            # FID / R-Precision 평가
            fid, top1, best_fid = evaluate_seg_vqdec(
                net, eval_loader, eval_wrapper, epoch, global_step, best_fid, device
            )
            if fid == best_fid:
                torch.save({'epoch': epoch, 'model': net.state_dict()},
                           pjoin(ckpt_dir, f'{s2_name}_best_fid.tar'))

        if (epoch + 1) % cfg.training.save_every == 0:
            torch.save({'epoch': epoch, 'model': net.state_dict()},
                       pjoin(ckpt_dir, f'{s2_name}_ep{epoch+1:04d}.tar'))
