import os
from os.path import join as pjoin

import torch

from model.transformer.moscale import MoScale
from model.vq.hrvqvae import HRVQVAE

from config.load_config import load_config

from utils.fixseeds import fixseed

from utils.motion_process import recover_from_ric
from utils.utils import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np

def inv_transform(data):
    return data * std + mean


def load_vq_model(vq_cfg, device):
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

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'hrvqvae', vq_cfg.exp.name, 'model', moscale_cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model


def load_trans_model(t2m_cfg, which_model, device):
    moscale = MoScale(
        code_dim=t2m_cfg.vq.code_dim,
        latent_dim=t2m_cfg.model.latent_dim,
        num_heads=t2m_cfg.model.n_heads,
        dropout=t2m_cfg.model.dropout,
        attn_drop_rate=t2m_cfg.model.attn_drop_rate,
        text_dim=t2m_cfg.text_embedder.dim_embed,
        cond_drop_prob=t2m_cfg.training.cond_drop_prob,
        device=device,
        cfg=t2m_cfg,
        full_length=t2m_cfg.data.max_motion_length//4,
        scales=[8, 4, 2, 1]
    )

    ckpt = torch.load(pjoin(t2m_cfg.exp.root_ckpt_dir, t2m_cfg.data.name, "moscale", t2m_cfg.exp.name, 'model', which_model),
                      map_location=device, weights_only=True)
    moscale.load_state_dict(ckpt["moscale"])

    moscale.to(device)
    moscale.eval()
    print(f'Loading MoScale {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    return moscale


def tokenize_motion(motion_np, m_length, vq_model, mean, std, device, max_len):
    """
    Normalize and tokenize a source motion.

    Args:
        motion_np: numpy array [1, T, 263] or [T, 263]
        m_length:  LongTensor [1], frame-level motion length (already aligned to unit_length)
        vq_model:  HRVQVAE
        mean, std: normalization statistics [263]
        device:    torch device
        max_len:   int, max_motion_length to pad to (CNN encoder needs consistent input size)

    Returns:
        source_tokens: list of len(scales) LongTensor [1, pl_i], VQ codes (-1 = padding)
    """
    if motion_np.ndim == 2:
        motion_np = motion_np[np.newaxis]   # [1, T, 263]

    T = m_length[0].item()
    # Pad to max_motion_length so the CNN encoder sees a consistent input size
    motion_padded = np.zeros((1, max_len, motion_np.shape[-1]), dtype=np.float32)
    motion_padded[0, :T] = motion_np[0, :T]

    motion_tensor = torch.tensor(motion_padded, device=device)
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
    std_t  = torch.tensor(std,  device=device, dtype=torch.float32)
    motion_norm = (motion_tensor - mean_t) / std_t

    with torch.no_grad():
        source_tokens, _, _ = vq_model.encode(
            motion_norm, m_length.clone().float())

    return source_tokens   # list of 4 tensors [1, pl_i]


if __name__ == '__main__':
    cfg = load_config("./config/edit.yaml")
    fixseed(cfg.seed)

    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)

    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    cfg.gen_dir = pjoin(cfg.checkpoint_dir, 'editing', cfg.ext)

    os.makedirs(cfg.gen_dir, exist_ok=True)
    os.makedirs(pjoin(cfg.gen_dir, 'bvh'), exist_ok=True)
    os.makedirs(pjoin(cfg.gen_dir, 'joints'), exist_ok=True)
    os.makedirs(pjoin(cfg.gen_dir, 'animations'), exist_ok=True)

    # Load models
    moscale_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name, 'train_moscale.yaml'))
    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'hrvqvae', moscale_cfg.vq_name, 'train_hrvqvae.yaml'))
    moscale_cfg.vq = vq_cfg.quantizer
    vq_model = load_vq_model(vq_cfg, device)

    mean = np.load(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'Comp_v6_KLD005/meta', 'mean.npy'))
    std  = np.load(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'Comp_v6_KLD005/meta', 'std.npy'))

    moscale = load_trans_model(moscale_cfg, cfg.which_epoch, device)

    ##### ---- Data ---- #####
    captions = [cfg.text_prompt]
    original_motion = np.load(cfg.source_motion)  # [T, 263] or [1, T, 263]
    if original_motion.ndim == 3:
        original_motion = original_motion[0]       # always work with [T, 263]

    # Align source length to unit_length (4 frames)
    T_frames = len(original_motion) // cfg.data.unit_length * cfg.data.unit_length - cfg.data.unit_length
    original_motion = original_motion[:T_frames]
    src_length = torch.LongTensor([T_frames]).to(device)

    # Target length: use cfg.target_length if set and > source, otherwise keep source length.
    # Set target_length > T_frames in the config to extend the motion beyond its original duration.
    target_length_cfg = cfg.get('target_length', [-1])[0]
    if target_length_cfg > 0 and target_length_cfg > T_frames:
        # Align to unit_length
        T_target = target_length_cfg // cfg.data.unit_length * cfg.data.unit_length
        T_target = min(T_target, cfg.data.max_motion_length)
    else:
        T_target = T_frames
    m_length = torch.LongTensor([T_target]).to(device)

    if T_target > T_frames:
        print(f"Extension mode: source {T_frames} frames → target {T_target} frames")
    else:
        print(f"Edit mode: source {T_frames} frames")

    # Parse edit region from config: mask_edit_section is a list of "start, end" strings.
    # Fractions are relative to the TARGET length.
    section_str = cfg.mask_edit_section[0]
    edit_start_frac, edit_end_frac = [float(x.strip()) for x in section_str.split(',')]
    print(f"Edit region: [{edit_start_frac:.2f}, {edit_end_frac:.2f}] of the target motion")

    # Tokenize source motion
    source_tokens = tokenize_motion(original_motion, src_length, vq_model, mean, std, device, cfg.data.max_motion_length)

    kinematic_chain = t2m_kinematic_chain

    for r in range(cfg.repeat_time):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            mids, f_hat = moscale.edit(
                captions,
                m_length // (2 ** vq_model.down_t),  # token-space target lengths
                source_tokens=source_tokens,
                edit_start_frac=edit_start_frac,
                edit_end_frac=edit_end_frac,
                cond_scale=cfg.cond_scales[0],
                temperature=cfg.temperature[0],
                vq_model=vq_model,
                sample_time=cfg.sample_times[0],
            )
            # Decode from blended f_hat rather than token indices to preserve
            # non-edit regions with zero re-quantization error.
            token_lens = m_length // (2 ** vq_model.down_t)
            pred_motions = vq_model.decoder(f_hat, token_lens)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = inv_transform(pred_motions)

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d" % (k, caption, m_length[k]))
            animation_path = pjoin(cfg.gen_dir, 'animations', str(k))
            bvh_path       = pjoin(cfg.gen_dir, 'bvh', str(k))
            joint_path     = pjoin(cfg.gen_dir, 'joints', str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(bvh_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy" % (k, r, m_length[k])), joint_data)
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), cfg.data.joint_num).numpy()

            pred_save_path = pjoin(animation_path, "sample%d_repeat%d_pred_len%d.mp4" % (k, r, m_length[k]))
            plot_3d_motion(pred_save_path, kinematic_chain, joint, title=caption, fps=cfg.data.fps, radius=4)

            # Visualise original motion for comparison
            original_joint = recover_from_ric(
                torch.from_numpy(original_motion.astype(np.float32)), cfg.data.joint_num).numpy()
            orig_save_path = pjoin(animation_path, "sample%d_repeat%d_pred_len%d_original.mp4" % (k, r, m_length[k]))
            plot_3d_motion(orig_save_path, kinematic_chain, original_joint, title=caption, fps=cfg.data.fps, radius=4)
