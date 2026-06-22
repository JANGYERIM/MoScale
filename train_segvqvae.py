import os
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
from utils.metrics import calculate_frechet_distance, calculate_R_precision_gpu, euclidean_distance_matrix_gpu, calculate_activation_statistics_gpu, calculate_diversity_gpu
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
import wandb


@torch.no_grad()
def evaluate_segvqvae(net, clip_tokenizer, clip_model, eval_loader,
                      eval_wrapper, ep, acc_iter, best_fid, device, clip_mean=None):
    """
    FID/Top1 평가: full caption을 N=1 segment로 취급해 SegVQVAE로 재구성 후 평가
    eval_loader: Text2MotionDatasetEval 기반 표준 로더
    """
    net.eval()
    clip_model.eval()

    motion_annotation_list, motion_pred_list = [], []
    R_prec_real = torch.zeros(3, device=device)
    R_prec_pred = torch.zeros(3, device=device)
    match_real = match_pred = nb_sample = 0.

    dataset = eval_loader.dataset
    std_gpu  = torch.from_numpy(dataset.std).float().to(device)
    mean_gpu = torch.from_numpy(dataset.mean).float().to(device)

    for batch in eval_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = batch
        motion   = motion.to(device)
        m_length = m_length.to(device)
        B = motion.shape[0]

        # full caption → CLIP → [B, 1, clip_dim]
        tok = clip_tokenizer(list(caption), padding=True, truncation=True,
                             max_length=77, return_tensors='pt').to(device)
        clip_embeds = clip_model(**tok).pooler_output  # [B, D]
        if clip_mean is not None:
            clip_embeds = F.normalize(clip_embeds - clip_mean.unsqueeze(0), dim=-1)
        clip_embeds = clip_embeds.unsqueeze(1)   # [B, 1, D]
        seg_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        x_recon, _, _ = net(motion, clip_embeds, seg_mask, m_length)

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

    div_real = calculate_diversity_gpu(em_all,   300 if nb_sample > 300 else 100)
    div      = calculate_diversity_gpu(pred_all, 300 if nb_sample > 300 else 100)

    top123_real = (R_prec_real / nb_sample).cpu().numpy()
    top123      = (R_prec_pred / nb_sample).cpu().numpy()
    match_real /= nb_sample
    match_pred /= nb_sample

    print(f"\n[Eval  Ep {ep:03d}] "
          f"FID {fid:.4f} | "
          f"Top1 {top123[0]:.4f} Top2 {top123[1]:.4f} Top3 {top123[2]:.4f} | "
          f"Div {div:.4f} | Match {match_pred:.4f}\n")

    wandb.log({'eval/FID':   fid,  'eval/Top1': top123[0],
               'eval/Top2':  top123[1], 'eval/Top3': top123[2],
               'eval/Div':   div,  'eval/Match': match_pred,
               'epoch': ep}, step=acc_iter)

    if fid < best_fid:
        print(f"  --> FID improved: {best_fid:.4f} → {fid:.4f}")
        best_fid = fid

    return fid, top123[0], best_fid


def compute_clip_mean(tokenizer, clip_model, jsonl_path, device, max_samples=5000):
    """
    학습 segment 텍스트 전체의 CLIP 임베딩 평균을 계산.
    centering 적용 시 cone bias를 제거해 embedding 간 구분력을 2배 향상.
    """
    import json
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
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tok = tokenizer(batch, padding=True, truncation=True,
                            max_length=77, return_tensors='pt').to(device)
            all_embeds.append(clip_model(**tok).pooler_output.cpu())
    mean_embed = torch.cat(all_embeds, dim=0).mean(0)  # [clip_dim]
    print(f"[CLIP centering] computed mean over {len(texts)} segment texts")
    return mean_embed.to(device)


def encode_segments(tokenizer, clip_model, seg_texts_batch, device, clip_mean=None):
    """
    seg_texts_batch: list of list of strings [B, max_n_seg]
    clip_mean: [clip_dim] tensor, subtract before normalizing to remove cone bias
    returns: clip_embeds [B, max_n_seg, clip_dim]
    """
    import torch.nn.functional as F
    B = len(seg_texts_batch)
    max_n = len(seg_texts_batch[0])
    flat_texts = [t for sample in seg_texts_batch for t in sample]

    with torch.no_grad():
        tokens = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(device)
        embeds = clip_model(**tokens).pooler_output  # [B*max_n, clip_dim]
        if clip_mean is not None:
            embeds = F.normalize(embeds - clip_mean.unsqueeze(0), dim=-1)
    return embeds.reshape(B, max_n, -1)


if __name__ == '__main__':
    cfg = load_config('config/train_segvqvae.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'segvqvae', cfg.exp.name)
    os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
    shutil.copy('config/train_segvqvae.yaml', cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)
    device = torch.device(cfg.exp.device)

    wandb.init(project='SegVQVAE', dir=cfg.exp.checkpoint_dir, config=dict(cfg), name=cfg.exp.name)

    # evaluator opt (mean/std/unit_length 등)
    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device, data_root=cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std  = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    wrapper_opt.motion_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')
    wrapper_opt.text_dir   = pjoin(cfg.data.root_dir, 'texts')

    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test',
                                               device=device, data_root=cfg.data.root_dir)

    # dataset
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

    # CLIP (frozen)
    clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.model.clip_model)
    clip_text_model = CLIPTextModel.from_pretrained(cfg.model.clip_model).to(device).eval()
    for p in clip_text_model.parameters():
        p.requires_grad = False

    # CLIP centering: subtract training-set mean to remove cone bias
    # cos-sim std doubles (0.10 → 0.19), mean shifts from 0.60 → ~0
    # 사전 계산: python scripts/compute_clip_mean.py
    clip_mean_path = 'config/clip_mean.pt'
    if os.path.exists(clip_mean_path):
        clip_mean = torch.load(clip_mean_path, map_location=device)
        print(f"[CLIP centering] loaded mean from {clip_mean_path}")
    else:
        clip_mean = compute_clip_mean(clip_tokenizer, clip_text_model,
                                      cfg.data.seg_jsonl_train, device)
        torch.save(clip_mean, clip_mean_path)
        print(f"[CLIP centering] saved mean to {clip_mean_path}")

    # model
    net = SegVQVAE(cfg).to(device)
    print(f"SegVQVAE params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

    ckpt_prefix = cfg.training.get('stage2_ckpt_name', 'net')

    best_val_loss = float('inf')
    best_fid      = float('inf')
    global_step   = 0

    for epoch in range(cfg.training.num_epochs):
        net.train()
        ep_recon = ep_commit_seg = ep_commit_hrv = ep_align = ep_total = 0.
        ep_perplexity = ep_n_active = 0.
        ep_steps = 0

        for motion, seg_texts, m_lens, seg_mask, n_valid in train_loader:
            motion   = motion.to(device)
            seg_mask = seg_mask.to(device)
            m_lens   = m_lens.to(device)

            seg_texts_list = [list(col) for col in zip(*seg_texts)]
            clip_embeds = encode_segments(clip_tokenizer, clip_text_model,
                                          seg_texts_list, device, clip_mean)

            _, loss, log = net(motion, clip_embeds, seg_mask, m_lens)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            ep_recon        += log['l_recon']
            ep_commit_seg   += log['l_commit_seg']
            ep_commit_hrv   += log['l_commit_hrv']
            ep_align        += log['l_align']
            ep_total        += loss.item()
            ep_perplexity   += log['vq_perplexity']
            ep_n_active     += log['vq_n_active']
            ep_steps        += 1

            print(f"Ep {epoch:03d} | step {global_step} | "
                  f"recon {log['l_recon']:.4f} | "
                  f"commit_seg {log['l_commit_seg']:.4f} | "
                  f"commit_hrv {log['l_commit_hrv']:.4f} | "
                  f"align {log['l_align']:.4f} | "
                  f"ppl {log['vq_perplexity']:.1f} | "
                  f"active {int(log['vq_n_active'])}/{net.vq.nb_code}")

            if global_step % cfg.training.log_every == 0:
                wandb.log({'train/loss_recon':      log['l_recon'],
                           'train/loss_commit_seg': log['l_commit_seg'],
                           'train/loss_commit_hrv': log['l_commit_hrv'],
                           'train/loss_align':      log['l_align'],
                           'train/vq_perplexity':   log['vq_perplexity'],
                           'train/vq_n_active':     log['vq_n_active'],
                           'epoch': epoch}, step=global_step)

            global_step += 1

        # epoch summary
        print(f"\n[Train Ep {epoch:03d}] "
              f"total {ep_total/ep_steps:.4f} | "
              f"recon {ep_recon/ep_steps:.4f} | "
              f"commit_seg {ep_commit_seg/ep_steps:.4f} | "
              f"commit_hrv {ep_commit_hrv/ep_steps:.4f} | "
              f"align {ep_align/ep_steps:.4f} | "
              f"ppl {ep_perplexity/ep_steps:.1f} | "
              f"active {ep_n_active/ep_steps:.1f}/{net.vq.nb_code}\n")
        wandb.log({'epoch/loss_total':      ep_total      / ep_steps,
                   'epoch/loss_recon':      ep_recon      / ep_steps,
                   'epoch/loss_commit_seg': ep_commit_seg / ep_steps,
                   'epoch/loss_commit_hrv': ep_commit_hrv / ep_steps,
                   'epoch/loss_align':      ep_align      / ep_steps,
                   'epoch/vq_perplexity':   ep_perplexity / ep_steps,
                   'epoch/vq_n_active':     ep_n_active   / ep_steps,
                   'epoch': epoch}, step=global_step)

        # validation
        if (epoch + 1) % cfg.training.val_every == 0:
            net.eval()
            val_recon = val_commit_seg = val_commit_hrv = val_align = val_total = 0.
            n_val = 0
            with torch.no_grad():
                for motion, seg_texts, m_lens, seg_mask, n_valid in val_loader:
                    motion   = motion.to(device)
                    seg_mask = seg_mask.to(device)
                    m_lens   = m_lens.to(device)
                    seg_texts_list = [list(col) for col in zip(*seg_texts)]
                    clip_embeds = encode_segments(clip_tokenizer, clip_text_model,
                                                  seg_texts_list, device, clip_mean)
                    _, loss, log = net(motion, clip_embeds, seg_mask, m_lens)
                    val_recon      += log['l_recon']
                    val_commit_seg += log['l_commit_seg']
                    val_commit_hrv += log['l_commit_hrv']
                    val_align      += log['l_align']
                    val_total      += loss.item()
                    n_val += 1

            val_recon      /= n_val
            val_commit_seg /= n_val
            val_commit_hrv /= n_val
            val_align      /= n_val
            val_total      /= n_val
            print(f"[Val   Ep {epoch:03d}] "
                  f"total {val_total:.4f} | "
                  f"recon {val_recon:.4f} | "
                  f"commit_seg {val_commit_seg:.4f} | "
                  f"commit_hrv {val_commit_hrv:.4f} | "
                  f"align {val_align:.4f}")
            wandb.log({'val/loss_total':      val_total,
                       'val/loss_recon':      val_recon,
                       'val/loss_commit_seg': val_commit_seg,
                       'val/loss_commit_hrv': val_commit_hrv,
                       'val/loss_align':      val_align,
                       'epoch': epoch}, step=global_step)

            if val_recon < best_val_loss:
                best_val_loss = val_recon
                torch.save({'epoch': epoch, 'model': net.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           pjoin(cfg.exp.checkpoint_dir, f'{ckpt_prefix}_best.tar'))
                print(f"  --> best model saved (recon={val_recon:.4f})")

            # FID / Top1 evaluation
            fid, top1, best_fid = evaluate_segvqvae(
                net, clip_tokenizer, clip_text_model, eval_loader,
                eval_wrapper, epoch, global_step, best_fid, device, clip_mean
            )
            if fid == best_fid:
                torch.save({'epoch': epoch, 'model': net.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           pjoin(cfg.exp.checkpoint_dir, f'{ckpt_prefix}_best_fid.tar'))

        if (epoch + 1) % cfg.training.save_every == 0:
            torch.save({'epoch': epoch, 'model': net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       pjoin(cfg.exp.checkpoint_dir, f'{ckpt_prefix}_ep{epoch+1:04d}.tar'))
