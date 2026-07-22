
import glob
import os
from os.path import join as pjoin

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model.vq.hrvqvae import HRVQVAE
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader

from model.transformer.moscale import MoScale
from model.flow_decoder.rectified_flow import RectifiedFlowDecoder
from model.flow_decoder.unet1d_backbone import Unet1DforFlowDecoder

from config.load_config import load_config

import utils.eval_t2m as eval_t2m
from utils.fixseeds import fixseed
from utils.get_opt import get_opt

import argparse
import numpy as np
import time

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

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'hrvqvae', vq_cfg.exp.name, 'model',moscale_cfg.vq_ckpt),
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
                      map_location=cfg.device, weights_only=True)
    moscale.load_state_dict(ckpt["moscale"])

    moscale.to(device)
    moscale.eval()
    print(f'Loading MoScale {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    print(f'Loading MoScale {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!', file=f, flush=True)
    return moscale


def load_flow_decoder(flow_cfg, vq_cfg, ckpt_name, device):
    """Rebuild the rectified-flow decoder from its own saved train_flow_decoder.yaml and load
    one of its checkpoints (net_best_fid.tar / net_best_mpjpe.tar). Used to decode MoScale's
    predicted token ids in place of HRVQVAE's own deterministic decoder."""
    u = flow_cfg.model.unet1d
    denoiser = Unet1DforFlowDecoder(
        dim=u.dim,
        dim_mults=u.dim_mults,
        resnet_per_block=u.resnet_per_block,
        c_in_dim=vq_cfg.quantizer.code_dim,
        c_proj_dim=u.c_proj_dim,
        up_conv_c=u.up_conv_c,
        channels=flow_cfg.data.dim_pose,
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

    ckpt_path = pjoin(flow_cfg.exp.root_ckpt_dir, flow_cfg.data.name, 'flow_decoder', flow_cfg.exp.name, 'model', ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    flow_model.load_state_dict(ckpt['flow_model'])
    flow_model.to(device)
    flow_model.eval()
    print(f'Loading flow decoder {flow_cfg.exp.name} from epoch {ckpt["ep"]} ({ckpt_name})')
    print(f'Loading flow decoder {flow_cfg.exp.name} from epoch {ckpt["ep"]} ({ckpt_name})', file=f, flush=True)
    return flow_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_decoder_name', type=str, default=None,
                         help='overrides cfg.flow_decoder_name; set to decode MoScale-predicted tokens '
                              'with a trained flow decoder instead of HRVQVAE\'s own decoder')
    parser.add_argument('--flow_decoder_ckpt', type=str, default=None, help='overrides cfg.flow_decoder_ckpt')
    args = parser.parse_args()

    cfg = load_config("./config/eval_moscale.yaml")
    if args.flow_decoder_name is not None:
        cfg.flow_decoder_name = args.flow_decoder_name
    if args.flow_decoder_ckpt is not None:
        cfg.flow_decoder_ckpt = args.flow_decoder_ckpt

    fixseed(cfg.seed)

    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)
    torch.autograd.set_detect_anomaly(True)

    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    cfg.eval_dir = pjoin(cfg.checkpoint_dir, 'eval')

    os.makedirs(cfg.eval_dir, exist_ok=True)

    use_flow_decoder = bool(cfg.get('flow_decoder_name'))
    if use_flow_decoder:
        ckpt_stem = os.path.splitext(cfg.flow_decoder_ckpt)[0]
        out_path = pjoin(cfg.eval_dir, f"{cfg.ext}_flowdec_{cfg.flow_decoder_name}_{ckpt_stem}.log")
    else:
        out_path = pjoin(cfg.eval_dir, "%s.log" % cfg.ext)

    f = open(pjoin(out_path), 'w')

    moscale_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'moscale', cfg.moscale_name, 'train_moscale.yaml'))

    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'hrvqvae', moscale_cfg.vq_name, 'train_hrvqvae.yaml'))
    moscale_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, device)

    flow_model = None
    if use_flow_decoder:
        flow_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'flow_decoder', cfg.flow_decoder_name)
        # Different train_flow_decoder*.py variants (e.g. train_flow_decoder.py vs
        # train_flow_decoder_predicted.py) each save their own config under their own script's
        # filename, not a single fixed name -- pick whichever train_flow_decoder*.yaml is there.
        candidates = sorted(glob.glob(pjoin(flow_dir, 'train_flow_decoder*.yaml')))
        assert candidates, f'No train_flow_decoder*.yaml found in {flow_dir}'
        flow_cfg = load_config(candidates[0])
        flow_model = load_flow_decoder(flow_cfg, vq_cfg, cfg.flow_decoder_ckpt, device)

    dataset_opt_path = 'checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'), data_root=cfg.data.root_dir)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    eval_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=device, data_root=cfg.data.root_dir)


    for file in os.listdir(cfg.model_dir):
        if cfg.which_epoch != "all" and cfg.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        moscale = load_trans_model(moscale_cfg, file, device)

        for sample_time in cfg.sample_times:
            for cs in cfg.cond_scales:
                for temp in cfg.temperature:
                    for top_p in cfg.top_p_thres:
                        fid = []
                        div = []
                        top1 = []
                        top2 = []
                        top3 = []
                        matching = []
                        mm = []
                        for i in range(cfg.repeat_time):
                            print(f'Sample time: {sample_time}, Guidance scale: {cs}, temperature: {temp}, top_p: {top_p}')
                            print(f'Sample time: {sample_time}, Guidance scale: {cs}, temperature: {temp}, top_p: {top_p}', file=f, flush=True)
                            print('begin timing (raw)')
                            begin = time.time()
                            with torch.no_grad():
                                best_fid, best_div, Rprecision, best_matching, best_mm = (
                                    eval_t2m.evaluation_moscale(
                                        eval_loader,
                                        vq_model,
                                        moscale,
                                        i,
                                        eval_wrapper=eval_wrapper,
                                        cond_scale=cs,
                                        cal_mm=cfg.cal_mm,
                                        sample_time=sample_time,
                                        temperature=temp,
                                        top_p_thres=top_p,
                                        flow_model=flow_model,
                                        dim_pose=cfg.data.dim_pose,
                                        ode_steps=cfg.get('flow_decoder_ode_steps', 16)
                                    )
                                )
                            end = time.time()
                            print(f'Evaluation time: {end-begin:.2f} seconds')
                            fid.append(best_fid)
                            div.append(best_div)
                            top1.append(Rprecision[0])
                            top2.append(Rprecision[1])
                            top3.append(Rprecision[2])
                            matching.append(best_matching)
                            mm.append(best_mm)

                        fid = np.array(fid)
                        div = np.array(div)
                        top1 = np.array(top1)
                        top2 = np.array(top2)
                        top3 = np.array(top3)
                        matching = np.array(matching)
                        mm = np.array(mm)

                        print(f'{file} final result (Guidance scale: {cs}, top_p: {top_p}):')
                        print(f'{file} final result (Guidance scale: {cs}, top_p: {top_p}):', file=f, flush=True)

                        msg_final = (
                            f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                            f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                            f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(cfg.repeat_time):.3f}, "
                            f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(cfg.repeat_time):.3f}, "
                            f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                            f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n"
                            f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(cfg.repeat_time):.3f}\n\n"
                        )
                        print(msg_final)
                        print(msg_final, file=f, flush=True)

    f.close()
