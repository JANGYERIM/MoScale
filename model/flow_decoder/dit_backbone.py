"""
DiT-style flow-velocity estimator for the rectified-flow motion decoder.
Ported from /nas2/data/dpfla3573/code/DisCoRD/MotionPriors/models/rf_decoder/DiTforflow_decoder.py
(itself adapted from https://github.com/facebookresearch/DiT), trimmed of the built-in
CLIP text conditioning path — MoScale already owns its own T5 encoder and text conditioning
is added in a later stage, not this base port.

NOT the active backbone: DisCoRD's own reference config (configs/config_model.yaml) defaults
to the Unet1D backbone (model/flow_decoder/unet1d_backbone.py), which is what
run/train_flow_decoder.py / config/train_flow_decoder.yaml actually wire up, for fidelity to
the paper. This DiT variant is kept as a working, MoScale-style alternative but is not
imported by the training entrypoint.
"""
import math

import torch
import torch.nn as nn

from model.flow_decoder.helpers import StackLinear, init_faceformer_biased_mask_future, make_temporal_mask


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Minimal Linear-act-Linear MLP (avoids a hard dependency on timm for one block)."""
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar flow-matching timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT1DBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, key_padding_mask=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        scaled_attn = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(scaled_attn, scaled_attn, scaled_attn,
                                    key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_output
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1D(nn.Module):
    """The final readout layer of DiT."""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT1DforFlowDecoder(nn.Module):
    """
    Flow-velocity estimator: predicts the rectified-flow velocity field for raw motion,
    conditioned on the HRVQVAE quantized latent `y` via a small upsampling "projection"
    (`cin_proj`) that repeats each latent frame `2**quant_factor` times (matching HRVQVAE's
    `down_t`) and mixes channels back down to `c_proj_dim` before per-frame conditioning.
    """
    def __init__(self, out_dim=263, embed_dim=384, c_in_dim=512, c_proj_dim=384,
                 num_heads=8, mlp_ratio=4, depth=6, max_seq_len=196, drop_out_prob=0.1,
                 quant_factor=2, temporal_bias="alibi_future"):
        super().__init__()
        self.out_dim = out_dim
        self.quant_factor = quant_factor
        self.upsample_factor = 2 ** quant_factor

        self.blocks = nn.ModuleList([
            DiT1DBlock(hidden_size=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=drop_out_prob)
            for _ in range(depth)
        ])
        self.in_proj = nn.Linear(out_dim + c_proj_dim, embed_dim)
        self.out_proj = FinalLayer1D(embed_dim, out_dim)

        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)

        self.cin_proj = nn.Sequential(
            StackLinear(quant_factor=quant_factor, seq_first=True, unstack=False),
            nn.Linear(c_in_dim * self.upsample_factor, c_in_dim * self.upsample_factor),
            StackLinear(quant_factor=quant_factor, seq_first=True, unstack=True),
            nn.Linear(c_in_dim, c_proj_dim),
        )

        if temporal_bias == "alibi_future":
            self.attn_mask = init_faceformer_biased_mask_future(num_heads, max_seq_len)
        else:
            self.attn_mask = None

    def forward(self, x, times, y, padding_mask=None):
        # x: noised motion (B, T, out_dim); times: (B,); y: HRVQVAE quantized latent (B, T//upsample_factor, c_in_dim)
        t = self.t_embedder(times)  # (B, embed_dim)

        y = y.repeat_interleave(self.upsample_factor, dim=1)  # (B, T, c_in_dim)
        y = self.cin_proj(y)  # (B, T, c_proj_dim)

        inputs = torch.cat([x, y], dim=-1)
        inputs = self.in_proj(inputs)  # (B, T, embed_dim)

        if self.attn_mask is not None:
            attn_mask = make_temporal_mask(inputs, self.attn_mask)
        else:
            attn_mask = None

        for block in self.blocks:
            inputs = block(inputs, t, key_padding_mask=padding_mask, attn_mask=attn_mask)

        return self.out_proj(inputs, t)
