import math

import torch
import torch.nn as nn


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


def init_faceformer_biased_mask_future(num_heads, max_seq_len, period=1):
    """
    Symmetric ALiBi-style attention bias (FaceFormer paper) with the future NOT masked out:
    attention decays the farther a key is from the query, in both directions.
    """
    slopes = torch.Tensor(get_slopes(num_heads))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // period
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = alibi + torch.flip(alibi, [1, 2])
    return mask


def make_temporal_mask(trans_inputs, attention_mask):  # trans_inputs: (B, T, C)
    mask = None
    B, T = trans_inputs.shape[:2]
    if attention_mask is not None:
        mask = attention_mask[:, :T, :T].clone().detach().to(device=trans_inputs.device)
        if mask.ndim == 3:  # needs to broadcast to num_head * batch_size
            mask = mask.repeat(B, 1, 1)
    return mask


class StackLinear(nn.Module):
    """(B, T, F) <-> (B, T // 2**quant_factor, F * 2**quant_factor) reshape, used around a
    channel-mixing projection to let it see `2**quant_factor` neighbouring latent frames at
    once before splitting back out to per-frame conditioning."""
    def __init__(self, quant_factor=2, unstack=False, seq_first=True):
        super().__init__()
        self.quant_factor = quant_factor
        self.latent_frame_size = 2 ** quant_factor
        self.unstack = unstack
        self.seq_first = seq_first

    def forward(self, x):
        if self.seq_first:
            B, T, F = x.shape
        else:
            B, F, T = x.shape
            x = x.permute(0, 2, 1)

        if not self.unstack:
            assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
            T_latent = T // self.latent_frame_size
            F_stack = F * self.latent_frame_size
            x = x.reshape(B, T_latent, F_stack)
        else:
            F_stack = F // self.latent_frame_size
            x = x.reshape(B, T * self.latent_frame_size, F_stack)

        if not self.seq_first:
            x = x.permute(0, 2, 1)

        return x
