import os
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import wandb
from model.flow_decoder.lr_scheduler import CosineAnnealingWarmupRestarts
from trainers.base_trainer import BaseTrainer
from utils.eval_t2m import evaluation_flow_decoder
from utils.utils import print_current_loss, print_val_loss


def def_value():
    return 0.0


class FlowDecoderTrainer(BaseTrainer):
    def __init__(self, cfg, flow_model, vq_model, device, precomputed_z=False):
        self.cfg = cfg
        self.device = device
        self.vq_model = vq_model
        self.vq_model.eval()

        # If True, `forward` expects batches of (motion, z) with `z` already computed offline
        # (e.g. from MoScale's own greedy predicted tokens -- see
        # dataset/predicted_token_dataset.py) instead of re-encoding the GT window through
        # `vq_model` on the fly.
        self.precomputed_z = precomputed_z

        # `self.model` is the raw (never DataParallel-wrapped) module -- single source of
        # truth for EMA, `.sample()` during eval, and checkpoint state_dicts. DataParallel
        # wraps it as `self.module`, which would prefix state_dict/named_parameters keys with
        # "module." and doesn't expose custom methods like `.sample()`.
        self.model = flow_model
        self.model.to(device)

        gpu_ids = cfg.training.get('gpu_ids', None)
        if gpu_ids and len(gpu_ids) > 1:
            print(f'[FlowDecoderTrainer] DataParallel across GPUs {gpu_ids}')
            self.flow_model = nn.DataParallel(self.model, device_ids=gpu_ids)
        else:
            self.flow_model = self.model

        self.ema_rate = cfg.training.get('ema_rate', None)
        if self.ema_rate:
            self.ema_model = deepcopy(self.model).to(device)
            self.ema_model.eval()
            self.requires_grad(self.ema_model, False)

        self.logger = SummaryWriter(cfg.exp.log_dir)

    def forward(self, batch_data):
        if self.precomputed_z:
            _, motion, z = batch_data
            motion = motion.detach().to(self.device).float()
            z = z.detach().to(self.device).float()
            padding_mask = None  # every window is exactly window_size frames, no padding
        else:
            _, motion, m_lens = batch_data
            motion = motion.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device).long()

            with torch.no_grad():
                z = self.vq_model.get_quantized_latent(motion[..., :self.cfg.data.dim_pose], m_lens.clone())
            z = z.permute(0, 2, 1)  # (B, code_dim, T_z) -> (B, T_z, code_dim)

            mask = self.lengths_to_mask(m_lens, max_len=motion.shape[1])  # True == valid frame
            padding_mask = ~mask  # True == padded frame (matches flow decoder / attn convention)

        # DataParallel runs one replica per GPU and returns one scalar loss per replica
        # (gathered into a small vector on the primary device); .mean() reduces that back to
        # a single scalar. On a single GPU this is already a 0-dim scalar, so .mean() is a
        # no-op.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.flow_model(motion[..., :self.cfg.data.dim_pose], y=z, padding_mask=padding_mask)
        return loss.mean()

    def save(self, file_name, ep):
        torch.save({'flow_model': self.model.state_dict(), 'ep': ep}, file_name)

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.model.to(self.device)
        self.vq_model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr,
                                      betas=(0.9, 0.99), weight_decay=self.cfg.training.get('weight_decay', 0.0))

        max_epoch = self.cfg.training.max_epoch
        if self.cfg.training.get('scheduler', None) == 'cosine_warmup':
            warmup_steps = min(max_epoch // 10, 50)
            self.scheduler = CosineAnnealingWarmupRestarts(
                self.optimizer, first_cycle_steps=max_epoch, max_lr=self.cfg.training.lr,
                min_lr=self.cfg.training.get('min_lr', 1e-5), warmup_steps=warmup_steps)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epoch)

        if self.ema_rate:
            self.update_ema(self.ema_model, self.model, decay=0)

        epoch = 0
        it = 0

        best_fid, best_mpjpe = 1000., 1000.

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.cfg.training.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        while epoch < self.cfg.training.max_epoch:
            self.model.train()
            self.vq_model.eval()

            for i, batch_data in enumerate(train_loader):
                it += 1
                loss = self.forward(batch_data)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                max_norm=self.cfg.training.get('max_grad_norm', 1.0))
                self.optimizer.step()

                if self.ema_rate:
                    self.update_ema(self.ema_model, self.model, decay=self.ema_rate)

                logs['loss'] += loss.item()
                logs['lr'] += self.optimizer.param_groups[0]['lr']

                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    wandb.log({f"Loss/{k}": v for k, v in mean_loss.items()}, step=it)

            self.scheduler.step()
            epoch += 1

            print('Validation time:')
            self.model.eval()
            val_logs = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for batch_data in val_loader:
                    loss = self.forward(batch_data)
                    val_logs['loss'] += loss.item()
            mean_val_loss = OrderedDict()
            for tag, value in val_logs.items():
                self.logger.add_scalar('Val/%s' % tag, value / len(val_loader), epoch)
                mean_val_loss[tag] = value / len(val_loader)
            print_val_loss(mean_val_loss, epoch)

            eval_model = self.ema_model if self.ema_rate else self.model
            torch.save({'flow_model': eval_model.state_dict(), 'ep': epoch},
                       os.path.join(self.cfg.exp.model_dir, 'latest.tar'))

            eval_every_e = self.cfg.training.get('eval_every_e', 5)
            is_last_epoch = epoch == self.cfg.training.max_epoch
            if epoch % eval_every_e == 0 or is_last_epoch:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fid, mpjpe = evaluation_flow_decoder(
                        self.cfg.exp.model_dir, eval_val_loader, eval_model, self.vq_model, self.cfg,
                        self.logger, epoch, it, best_fid=best_fid, best_mpjpe=best_mpjpe, eval_wrapper=eval_wrapper)
                best_fid, best_mpjpe = min(best_fid, fid), min(best_mpjpe, mpjpe)
