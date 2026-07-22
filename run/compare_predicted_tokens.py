import argparse
import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

from config.load_config import load_config
from dataset.humanml3d_dataset import Text2MotionDataset
from dataset.predicted_token_dataset import export_token_comparison_json
from model.transformer.moscale import MoScale
from model.vq.hrvqvae import HRVQVAE
from utils.fixseeds import *
from utils.get_opt import get_opt


def load_vq_model(cfg, device):
    vq_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'hrvqvae', cfg.vq_name, 'train_hrvqvae.yaml'))

    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)

    ckpt = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'hrvqvae', cfg.vq_name, 'model', cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_cfg


def load_trans_model(cfg, vq_cfg, device):
    moscale_cfg = load_config(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name, 'train_moscale.yaml'))
    moscale_cfg.vq = vq_cfg.quantizer

    moscale = MoScale(
        code_dim=moscale_cfg.vq.code_dim,
        latent_dim=moscale_cfg.model.latent_dim,
        num_heads=moscale_cfg.model.n_heads,
        dropout=moscale_cfg.model.dropout,
        attn_drop_rate=moscale_cfg.model.attn_drop_rate,
        text_dim=moscale_cfg.text_embedder.dim_embed,
        cond_drop_prob=moscale_cfg.training.cond_drop_prob,
        device=device,
        cfg=moscale_cfg,
        full_length=moscale_cfg.data.max_motion_length // 4,
        scales=[8, 4, 2, 1],
    )
    ckpt = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name, 'model', cfg.moscale_ckpt),
                       map_location=device, weights_only=True)
    moscale.load_state_dict(ckpt['moscale'])
    moscale.to(device)
    moscale.eval()
    for p in moscale.parameters():
        p.requires_grad = False
    print(f'Loading MoScale {cfg.moscale_name} from epoch {ckpt["ep"]} (frozen)')
    return moscale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_flow_decoder_predicted.yaml',
                         help='reuses vq_name/moscale_name/cond_scale/sample_time/top_p_thres from this config')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=50, help='number of clips to compare; None/0 = all')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_path', type=str, default=None,
                         help='defaults to checkpoint_dir/humanml3d/flow_decoder/<exp.name>/token_comparison_<split>.json')
    args = parser.parse_args()

    cfg = load_config(args.config)
    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)
    device = torch.device(cfg.exp.device)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), data_root=cfg.data.root_dir)
    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    vq_model, vq_cfg = load_vq_model(cfg, device=device)
    trans = load_trans_model(cfg, vq_cfg, device=device)

    split_file = pjoin(cfg.data.root_dir, f'{args.split}.txt')
    text_motion_dataset = Text2MotionDataset(wrapper_opt, mean, std, split_file)

    out_path = args.out_path
    if out_path is None:
        out_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'flow_decoder', cfg.exp.name, 'eval')
        os.makedirs(out_dir, exist_ok=True)
        out_path = pjoin(out_dir, f'token_comparison_{args.split}.json')

    max_samples = args.num_samples if args.num_samples else None
    export_token_comparison_json(text_motion_dataset, trans, vq_model, cfg.cond_scale, device, out_path,
                                  batch_size=args.batch_size, sample_time=cfg.get('sample_time', None),
                                  top_p_thres=cfg.get('top_p_thres', 0.9), max_samples=max_samples)
