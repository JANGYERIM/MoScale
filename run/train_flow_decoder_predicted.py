import os
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from config.load_config import load_config
from dataset.humanml3d_dataset import Text2MotionDataset
from dataset.predicted_token_dataset import make_predicted_token_dataset, PredictedTokenWindowDataset
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.flow_decoder.rectified_flow import RectifiedFlowDecoder
from model.flow_decoder.unet1d_backbone import Unet1DforFlowDecoder
from model.transformer.moscale import MoScale
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

    ckpt = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'hrvqvae', cfg.vq_name, 'model', cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model, vq_cfg


def load_trans_model(cfg, vq_cfg, device):
    """Loads the already-trained MoScale transformer named in cfg.moscale_name and freezes it --
    it is only ever used to build the predicted-token training data, never fine-tuned here."""
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
    cfg = load_config('config/train_flow_decoder_predicted.yaml')
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'flow_decoder', cfg.exp.name)

    wandb.init(project=cfg.exp.get('wandb_project', 'snap-Trans-hml-local'), dir=cfg.exp.checkpoint_dir,
               config=dict(cfg), name=cfg.exp.name)

    os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
    shutil.copy('config/train_flow_decoder_predicted.yaml', cfg.exp.checkpoint_dir)

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

    train_split_file = pjoin(cfg.data.root_dir, 'train.txt')
    val_split_file = pjoin(cfg.data.root_dir, 'val.txt')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), data_root=cfg.data.root_dir)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

    vq_model, vq_cfg = load_vq_model(cfg, device=device)
    trans = load_trans_model(cfg, vq_cfg, device=device)

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
    print('Total parameters of flow decoder: {}M'.format(pc / 1000_000))
    print(device)

    trainer = FlowDecoderTrainer(cfg, flow_model, vq_model=vq_model, device=device, precomputed_z=True)

    # ---- One-time precompute: greedy (top_k=1), iterative predicted tokens from the frozen
    # MoScale transformer, decoded into the flow decoder's z condition and paired with GT motion.
    # This is what makes training match what the flow decoder actually sees at real inference. ----
    window_size = cfg.data.get('window_size', 64)
    precompute_bs = cfg.training.get('precompute_batch_size', 64)

    train_text_motion = Text2MotionDataset(wrapper_opt, mean, std, train_split_file)
    val_text_motion = Text2MotionDataset(wrapper_opt, mean, std, val_split_file)

    # GT-vs-predicted token ids for every clip actually used to build this training set (not just
    # an ad-hoc sample -- see run/compare_predicted_tokens.py for that). Off by default since it
    # doubles vq_model.encode calls over the full train split; flip on in the config to inspect.
    export_tokens = cfg.get('export_token_json', False)
    token_json_dir = pjoin(cfg.exp.checkpoint_dir, 'eval')
    if export_tokens:
        os.makedirs(token_json_dir, exist_ok=True)

    train_gt, train_z, train_len = make_predicted_token_dataset(
        train_text_motion, trans, vq_model, cfg.cond_scale, precompute_bs, device,
        sample_time=cfg.get('sample_time', None), top_p_thres=cfg.get('top_p_thres', 0.9),
        export_json_path=pjoin(token_json_dir, 'token_comparison_train.json') if export_tokens else None)
    val_gt, val_z, val_len = make_predicted_token_dataset(
        val_text_motion, trans, vq_model, cfg.cond_scale, precompute_bs, device,
        sample_time=cfg.get('sample_time', None), top_p_thres=cfg.get('top_p_thres', 0.9),
        export_json_path=pjoin(token_json_dir, 'token_comparison_val.json') if export_tokens else None)

    train_dataset = PredictedTokenWindowDataset(train_gt, train_z, train_len, mean, std, window_size=window_size)
    eval_dataset = PredictedTokenWindowDataset(val_gt, val_z, val_len, mean, std, window_size=window_size)
    del train_text_motion, val_text_motion, train_gt, train_z, val_gt, val_z

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(eval_dataset, batch_size=cfg.training.val_batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)

    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device, data_root=cfg.data.root_dir)

    trainer.train(train_loader, val_loader, eval_loader, eval_wrapper)
