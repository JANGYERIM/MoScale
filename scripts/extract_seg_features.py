"""
Stage 1+2 완료 후 실행: retrieval database 구축
각 학습 샘플에 대해:
  - seg_feats [n_valid, D] : temporal split avg pool (scale0 source)
  - m_proj    [n_valid, A] : align_proj_motion 거친 정규화 벡터 (retrieval key)

저장:
  config/retrieval_db_train.pt
  config/retrieval_db_val.pt

테스트 시:
  t_proj = align_proj_text(CLIP(text))
  → cos_sim(t_proj, db[i]['m_proj']) → argmax → db[i]['seg_feats'] 가져와 scale0 사용
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from os.path import join as pjoin
from tqdm import tqdm

from model.vq.seg_vqvae import SegVQVAE
from dataset.seg_dataset import SegMotionDataset, seg_collate_fn
from config.load_config import load_config
from utils.get_opt import get_opt


@torch.no_grad()
def extract(net, loader, device):
    db = []
    for batch in tqdm(loader, desc='extracting'):
        motion, seg_texts, m_lens, seg_mask, n_valid = batch
        motion   = motion.to(device)
        m_lens   = m_lens.to(device).long()
        seg_mask = seg_mask.to(device)

        feat        = net.encode(motion, m_lens)              # [B, T/4, D]
        m_lens_down = m_lens // (2 ** net.down_t)

        # temporal split → seg_feats [B, N, D]
        seg_feats = net._temporal_seg_pool(feat, seg_mask, m_lens_down)

        # retrieval key: align_proj_motion → L2 normalize [B, N, A]
        m_proj = F.normalize(net.align_proj_motion(seg_feats), dim=-1)

        B = feat.shape[0]
        for i in range(B):
            nv = int(n_valid[i].item()) if hasattr(n_valid[i], 'item') else int(n_valid[i])
            db.append({
                'seg_feats': seg_feats[i, :nv].cpu(),   # [n_valid, D]
                'm_proj':    m_proj[i, :nv].cpu(),       # [n_valid, A]  retrieval key
                'n_valid':   nv,
            })
    return db


if __name__ == '__main__':
    cfg    = load_config('config/train_segvqvae.yaml')
    device = torch.device(cfg.exp.device)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, device, data_root=cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std  = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))
    wrapper_opt.motion_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')
    wrapper_opt.text_dir   = pjoin(cfg.data.root_dir, 'texts')

    net = SegVQVAE(cfg).to(device).eval()

    # Stage 2 best checkpoint 로드 (encoder + align projectors + quantizer + decoder 모두 포함)
    ckpt_dir  = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'segvqvae', cfg.exp.name)
    ckpt_path = pjoin(ckpt_dir, 'stage2_best.tar')
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model'])
    print(f"[Loaded] {ckpt_path}")

    for split in ['train', 'val']:
        jsonl = cfg.data.seg_jsonl_train if split == 'train' else cfg.data.seg_jsonl_val
        dataset = SegMotionDataset(
            wrapper_opt, mean, std,
            split_file=pjoin(cfg.data.root_dir, f'{split}.txt'),
            seg_jsonl_path=jsonl,
            max_n_seg=cfg.data.max_n_seg,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=8, drop_last=False,
                            collate_fn=seg_collate_fn)

        db = extract(net, loader, device)
        out_path = f'config/retrieval_db_{split}.pt'
        torch.save(db, out_path)
        print(f"[{split}] {len(db)} entries saved → {out_path}")
        print(f"  seg_feats shape example: {db[0]['seg_feats'].shape}")
        print(f"  m_proj    shape example: {db[0]['m_proj'].shape}")
