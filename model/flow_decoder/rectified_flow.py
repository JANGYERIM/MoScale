"""
Rectified-flow motion decoder wrapper.
Ported (trimmed of immiscible-flow, consistency-FM and LPIPS-loss options — unused for this
base pass) from
/nas2/data/dpfla3573/code/DisCoRD/MotionPriors/models/rf_decoder/rectified_flow.py
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchdiffeq import odeint

from model.flow_decoder.helpers import default, exists


class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, padding_mask=None, **kwargs):
        if padding_mask is None:
            return F.mse_loss(pred, target)
        loss = F.mse_loss(pred, target, reduction='none')
        mask = (~padding_mask).unsqueeze(-1).float()  # padding_mask True == padded frame
        return (loss * mask).sum() / (mask.sum() * pred.shape[-1] + 1e-8)


class RectifiedFlowDecoder(nn.Module):
    """
    Learns the velocity field that rectifies Gaussian noise into raw motion, conditioned on
    a per-frame projected latent `y` (the HRVQVAE quantized latent, upsampled by the DiT
    backbone's `cin_proj`). At inference, `sample()` ODE-integrates from noise to data.
    """
    def __init__(
        self,
        model: nn.Module,
        odeint_kwargs: dict = dict(atol=1e-5, rtol=1e-5, method='midpoint'),
        data_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.net = model
        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape
        self.loss_fn = MaskedMSELoss()

    def predict_flow(self, model, noised, *, times, y, padding_mask=None):
        batch = noised.shape[0]
        if times.numel() == 1:
            times = times.repeat(batch)
        return model(noised, times=times, y=y, padding_mask=padding_mask)

    @torch.no_grad()
    def sample(
        self,
        y,
        batch_size=1,
        steps=16,
        noise=None,
        padding_mask=None,
        data_shape: Tuple[int, ...] | None = None,
    ):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least once'

        was_training = self.training
        self.eval()

        def ode_fn(t, x):
            return self.predict_flow(self.net, x, times=t, y=y, padding_mask=padding_mask)

        # Device is taken from `y` (always passed in), not from `self.net.parameters()`: under
        # nn.DataParallel, replicated submodules detach their parameters from `_parameters`
        # (PyTorch sets them as plain, non-Parameter attributes on replicas), so
        # `next(self.net.parameters())` raises StopIteration on every replica. Reading the
        # device off an input tensor works regardless of DataParallel-wrapping.
        noise = default(noise, torch.randn((batch_size, *data_shape), device=y.device))
        times = torch.linspace(0., 1., steps, device=y.device)

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)
        sampled_data = trajectory[-1]

        self.train(was_training)
        return sampled_data

    def forward(self, data: Tensor, y, padding_mask=None, noise: Tensor | None = None):
        batch, *data_shape = data.shape
        self.data_shape = default(self.data_shape, data_shape)

        noise = default(noise, torch.randn_like(data))

        times = torch.rand(batch, device=data.device)
        padded_times = times.view(batch, *((1,) * (data.ndim - 1)))

        # x0 = noise (t=0), x1 = data (t=1); linear interpolation + straight-line target velocity
        noised = padded_times * data + (1. - padded_times) * noise
        flow = data - noise

        pred_flow = self.predict_flow(self.net, noised, times=times, y=y, padding_mask=padding_mask)

        return self.loss_fn(pred_flow, flow, padding_mask=padding_mask)
