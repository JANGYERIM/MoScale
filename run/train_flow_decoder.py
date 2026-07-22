import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from config.load_config import load_config
from dataset.humanml3d_dataset import MotionWindowDataset
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.flow_decoder.rectified_flow import RectifiedFlowDecoder
from model.flow_decoder.unet1d_backbone import Unet1DforFlowDecoder
from model.vq.hrvqvae import HRVQVAE
from trainers.flow_decoder_trainer import FlowDecoderTrainer
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

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'hrvqvae', vq_cfg.exp.name, 'model', cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_cfg


if __name__ == "__main__":
    cfg = load_config('config/train_flow_decoder.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'flow_decoder', cfg.exp.name)

    wandb.init(project=cfg.exp.get('wandb_project', 'snap-Trans-hml-local'), dir=cfg.exp.checkpoint_dir,
               config=dict(cfg), name=cfg.exp.name)

    os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
    shutil.copy('config/train_flow_decoder.yaml', cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)

    if cfg.exp.device != 'cpu':
        torch.cuda.set_device(cfg.exp.device)

    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, 'model')
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'flow_decoder', cfg.exp.name)

    os.makedirs(cfg.exp.model_dir, exist_ok=True)
    os.makedirs(cfg.exp.log_dir, exist_ok=True)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'new_joint_vecs')

    train_cid_split_file = pjoin(cfg.data.root_dir, 'train.txt')
    val_cid_split_file = pjoin(cfg.data.root_dir, 'val.txt')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), data_root=cfg.data.root_dir)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    vq_model, vq_cfg = load_vq_model(cfg, device=device)

    u = cfg.model.unet1d
    denoiser = Unet1DforFlowDecoder(
        dim=u.dim,
        dim_mults=u.dim_mults,
        resnet_per_block=u.resnet_per_block,
        c_in_dim=vq_cfg.quantizer.code_dim,
        c_proj_dim=u.c_proj_dim,
        up_conv_c=u.up_conv_c,
        channels=cfg.data.dim_pose,
        dropout=u.dropout,
        use_attention=u.use_attention,
        learned_sinusoidal_cond=u.learned_sinusoidal_cond,
        random_fourier_features=u.random_fourier_features,
        learned_sinusoidal_dim=u.learned_sinusoidal_dim,
        sinusoidal_pos_emb_theta=u.sinusoidal_pos_emb_theta,
        attn_dim_head=u.attn_dim_head,
        attn_heads=u.attn_heads,
    )
    flow_model = RectifiedFlowDecoder(model=denoiser)

    pc = sum(param.numel() for param in flow_model.parameters())
    print(flow_model)
    print('Total parameters of flow decoder: {}M'.format(pc / 1000_000))
    print(device)

    trainer = FlowDecoderTrainer(cfg, flow_model, vq_model=vq_model, device=device)

    window_size = cfg.data.get('window_size', 64)
    train_dataset = MotionWindowDataset(wrapper_opt, mean, std, train_cid_split_file, window_size=window_size)
    eval_dataset = MotionWindowDataset(wrapper_opt, mean, std, val_cid_split_file, window_size=window_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(eval_dataset, batch_size=cfg.training.val_batch_size, drop_last=True, num_workers=8,
                              shuffle=True, pin_memory=True)

    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device, data_root=cfg.data.root_dir)

    trainer.train(train_loader, val_loader, eval_loader, eval_wrapper)
