"""Multicoil M4Raw AENet training smoke test.

The AENet update is implemented at the coil-combined image level: a proposal
image is converted to per-coil k-space through normalized sensitivity maps,
measured k-space entries are replaced, and the result is converted back with
the adjoint operator.  The prepared M4Raw ``.pt`` file stores one complex
coil-combined ground truth per slice plus per-slice sensitivity maps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_tpami import (
    ComplexPrior,
    ResidualBlock,
    count_parameters,
    fft2c,
    ifft2c,
    loss_fn as base_loss_fn,
    metric_batch,
    seed_everything,
    to_2ch,
    to_complex,
)
from external_adapter import OfficialHUMUSNetAdapter, OfficialVarNetAdapter


ROOT = Path(__file__).resolve().parent


def training_loss_fn(
    recon: torch.Tensor,
    gt: torch.Tensor,
    trajectory,
    deep_supervision: float,
    nmse_weight: float,
) -> torch.Tensor:
    loss = base_loss_fn(recon, gt, trajectory, deep_supervision)
    if nmse_weight > 0:
        recon_mag = recon.abs().flatten(1)
        gt_mag = gt.abs().flatten(1)
        nmse = (recon_mag - gt_mag).square().sum(1) / gt_mag.square().sum(
            1
        ).clamp_min(1e-12)
        loss = loss + nmse_weight * nmse.mean()
    return loss


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    if ema_state.keys() != model_state.keys():
        raise RuntimeError("EMA model and training model have different state keys")
    for key in model_state:
        ema_value = ema_state[key]
        model_value = model_state[key]
        if ema_value.dtype.is_floating_point:
            ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def multicoil_forward(z: torch.Tensor, smap: torch.Tensor) -> torch.Tensor:
    """Map a coil-combined image to per-coil k-space."""
    return fft2c(z[:, None, :, :] * smap)


def multicoil_adjoint(k: torch.Tensor, smap: torch.Tensor) -> torch.Tensor:
    """Map per-coil k-space back to a coil-combined image."""
    return (torch.conj(smap) * ifft2c(k)).sum(dim=1)


def data_consistency(
    z: torch.Tensor,
    y: torch.Tensor,
    smap: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """One-step coil-combined DC with normalized sensitivity maps."""
    k = multicoil_forward(z, smap)
    kdc = y + (1.0 - mask[None, None]) * k
    return multicoil_adjoint(kdc, smap)


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Represent complex (B, C, H, W) as real (B, 2C, H, W)."""
    realimag = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
    return realimag.reshape(x.shape[0], 2 * x.shape[1], x.shape[2], x.shape[3])


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Inverse of complex_to_channels."""
    batch, two_c, height, width = x.shape
    coils = two_c // 2
    realimag = x.view(batch, coils, 2, height, width).permute(0, 1, 3, 4, 2)
    return torch.view_as_complex(realimag.contiguous())


def group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Residual double-convolution block used by the multiscale U-Net."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_ch), out_ch)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class ComplexUNet(nn.Module):
    """Multiscale complex residual prior with a bounded residual output."""

    def __init__(
        self,
        in_ch: int = 4,
        hidden: int = 32,
        depth: int = 3,
        out_ch: int = 2,
        out_scale: float = 0.2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.out_scale = out_scale
        self.in_conv = ConvBlock(in_ch, hidden)
        self.downs = nn.ModuleList()
        channels = [hidden]
        ch = hidden
        for _ in range(depth):
            next_ch = ch * 2
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, next_ch, 3, stride=2, padding=1),
                    nn.GroupNorm(group_count(next_ch), next_ch),
                    nn.SiLU(inplace=True),
                    ConvBlock(next_ch, next_ch),
                )
            )
            ch = next_ch
            channels.append(ch)
        self.ups = nn.ModuleList()
        for i in range(depth):
            skip_ch = channels[-2 - i]
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(ch, skip_ch, kernel_size=1),
                        ConvBlock(skip_ch * 2, skip_ch),
                    ]
                )
            )
            ch = skip_ch
        self.out = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for (reduce, block), skip in zip(self.ups, reversed(skips[:-1])):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = reduce(x)
            x = block(torch.cat((x, skip), dim=1))
        return self.out_scale * self.out(x)


class RecurrentResidualPrior(nn.Module):
    """Residual CNN prior with a recurrent hidden state channel."""

    def __init__(
        self,
        in_ch: int = 4,
        hidden: int = 32,
        depth: int = 6,
        out_ch: int = 2,
        out_scale: float = 0.2,
    ):
        super().__init__()
        self.out_scale = out_scale
        self.in_proj = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(depth)]
        )
        self.mid = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, padding=1),
            nn.GroupNorm(group_count(hidden * 2), hidden * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = self.blocks(h)
        h = self.mid(h)
        return self.out_scale * self.out(h)


class KSpaceResidualPrior(nn.Module):
    """Residual CNN that predicts corrections on missing k-space entries."""

    def __init__(
        self,
        coils: int = 4,
        hidden: int = 32,
        depth: int = 4,
        out_scale: float = 0.2,
    ):
        super().__init__()
        self.out_scale = out_scale
        self.in_proj = nn.Conv2d(4 * coils + 1, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(depth)]
        )
        self.out = nn.Conv2d(hidden, 2 * coils, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, k_prop: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        mask_map = mask[None, None].expand(k_prop.shape[0], 1, -1, -1)
        x = torch.cat(
            (
                complex_to_channels(k_prop),
                complex_to_channels(y),
                mask_map,
            ),
            dim=1,
        )
        h = F.silu(self.in_proj(x))
        h = self.blocks(h)
        return self.out_scale * self.out(h)


class SmapRefiner(nn.Module):
    """Learn a residual correction to fixed sensitivity maps."""

    def __init__(
        self,
        coils: int = 4,
        hidden: int = 32,
        depth: int = 3,
        out_scale: float = 0.1,
    ):
        super().__init__()
        self.out_scale = out_scale
        self.in_proj = nn.Conv2d(4 * coils + 1, hidden, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(depth)]
        )
        self.out = nn.Conv2d(hidden, 2 * coils, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, smap: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        coil_images = ifft2c(y)
        rss = torch.sqrt(
            torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=True)
        )
        x = torch.cat(
            (
                complex_to_channels(smap),
                complex_to_channels(coil_images),
                rss,
            ),
            dim=1,
        )
        h = F.silu(self.in_proj(x))
        h = self.blocks(h)
        delta = channels_to_complex(self.out_scale * self.out(h))
        refined = smap + delta
        norm = torch.sqrt(
            torch.sum(torch.abs(refined) ** 2, dim=1, keepdim=True)
        ).clamp_min(1e-6)
        return refined / norm


class SmapRefinedModel(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        coils: int = 4,
        hidden: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.base = base
        self.refiner = SmapRefiner(coils=coils, hidden=hidden, depth=depth)

    def forward(self, z0, y, smap, return_trajectory=False):
        refined = self.refiner(smap, y)
        return self.base(
            z0,
            y,
            refined,
            return_trajectory=return_trajectory,
        )


class AENetMC(nn.Module):
    """Shared anchored AENet with a multicoil data-consistency projection."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 8,
        hidden: int = 32,
        depth: int = 4,
        prior_factory=ComplexPrior,
        anchor_init: float = 0.8,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = prior_factory(in_ch=4, hidden=hidden, depth=depth)
        self.step_scale = nn.Parameter(torch.ones(steps, 2))
        self.step_bias = nn.Parameter(torch.zeros(steps, 2))
        self.log_step = nn.Parameter(torch.full((steps,), np.log(np.expm1(0.5))))
        self.log_anchor = nn.Parameter(
            torch.full((steps,), np.log(np.expm1(anchor_init)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta_2ch = self.prior(inp)
            delta_2ch = (
                delta_2ch * self.step_scale[k].view(1, 2, 1, 1)
                + self.step_bias[k].view(1, 2, 1, 1)
            )
            delta = to_complex(delta_2ch)
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            anchor = F.softplus(self.log_anchor[k]).clamp(0.05, 2.0)
            proposal = z + step * delta + anchor * (z0 - z)
            z = data_consistency(proposal, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class RecurrentAENetMC(nn.Module):
    """AENet with a recurrent image state carried across solver iterations."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 3,
        anchor_init: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.hidden = hidden
        self.prior = ComplexUNet(
            in_ch=4 + hidden,
            hidden=hidden,
            depth=depth,
            out_ch=2 + hidden,
            out_scale=0.1,
        )
        self.gate = nn.Conv2d(2 * hidden, hidden, kernel_size=1)
        self.step_scale = nn.Parameter(torch.ones(steps, 2))
        self.step_bias = nn.Parameter(torch.zeros(steps, 2))
        self.log_step = nn.Parameter(torch.full((steps,), np.log(np.expm1(0.5))))
        self.log_anchor = nn.Parameter(
            torch.full((steps,), np.log(np.expm1(anchor_init)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        h = torch.zeros(
            z.shape[0],
            self.hidden,
            z.shape[-2],
            z.shape[-1],
            device=z.device,
            dtype=z.real.dtype,
        )
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0), h), dim=1)
            out = self.prior(inp)
            delta_2ch = out[:, :2]
            h_new = out[:, 2:]
            gate = torch.sigmoid(self.gate(torch.cat((h, h_new), dim=1)))
            h = (1.0 - gate) * h + gate * torch.tanh(h_new)
            delta_2ch = (
                delta_2ch * self.step_scale[k].view(1, 2, 1, 1)
                + self.step_bias[k].view(1, 2, 1, 1)
            )
            delta = to_complex(delta_2ch)
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            anchor = F.softplus(self.log_anchor[k]).clamp(0.0, 1.0)
            proposal = z + step * delta + anchor * (z0 - z)
            z = data_consistency(proposal, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class RecurrentCNNAENetMC(nn.Module):
    """AENet with recurrent state over the stronger residual CNN prior."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 6,
        anchor_init: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.hidden = hidden
        self.prior = RecurrentResidualPrior(
            in_ch=4 + hidden,
            hidden=hidden,
            depth=depth,
            out_ch=2 + hidden,
            out_scale=0.1,
        )
        self.gate = nn.Conv2d(2 * hidden, hidden, kernel_size=1)
        self.step_scale = nn.Parameter(torch.ones(steps, 2))
        self.step_bias = nn.Parameter(torch.zeros(steps, 2))
        self.log_step = nn.Parameter(torch.full((steps,), np.log(np.expm1(0.5))))
        self.log_anchor = nn.Parameter(
            torch.full((steps,), np.log(np.expm1(anchor_init)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        h = torch.zeros(
            z.shape[0],
            self.hidden,
            z.shape[-2],
            z.shape[-1],
            device=z.device,
            dtype=z.real.dtype,
        )
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0), h), dim=1)
            out = self.prior(inp)
            delta_2ch = out[:, :2]
            h_new = out[:, 2:]
            gate = torch.sigmoid(self.gate(torch.cat((h, h_new), dim=1)))
            h = (1.0 - gate) * h + gate * torch.tanh(h_new)
            delta_2ch = (
                delta_2ch * self.step_scale[k].view(1, 2, 1, 1)
                + self.step_bias[k].view(1, 2, 1, 1)
            )
            delta = to_complex(delta_2ch)
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            anchor = F.softplus(self.log_anchor[k]).clamp(0.0, 1.0)
            proposal = z + step * delta + anchor * (z0 - z)
            z = data_consistency(proposal, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class AdaptiveRecurrentCNNAENetMC(nn.Module):
    """Recurrent CNN AENet with input-adaptive step and anchor maps."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 6,
        anchor_init: float = 0.05,
    ):
        super().__init__()
        if not 0.0 < anchor_init < 1.0:
            raise ValueError("anchor_init must be in (0, 1)")
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.hidden = hidden
        self.prior = RecurrentResidualPrior(
            in_ch=4 + hidden,
            hidden=hidden,
            depth=depth,
            out_ch=2 + hidden,
            out_scale=0.1,
        )
        self.gate = nn.Conv2d(2 * hidden, hidden, kernel_size=1)
        self.adapt = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.adapt[-1].weight)
        nn.init.zeros_(self.adapt[-1].bias)
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.anchor_logit = nn.Parameter(
            torch.full((steps,), math.log(anchor_init / (1.0 - anchor_init)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        h = torch.zeros(
            z.shape[0],
            self.hidden,
            z.shape[-2],
            z.shape[-1],
            device=z.device,
            dtype=z.real.dtype,
        )
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0), h), dim=1)
            out = self.prior(inp)
            delta_2ch = out[:, :2]
            h_new = out[:, 2:]
            gate = torch.sigmoid(self.gate(torch.cat((h, h_new), dim=1)))
            h = (1.0 - gate) * h + gate * torch.tanh(h_new)
            adaptive = self.adapt(h)
            step_map = F.softplus(
                self.log_step[k] + adaptive[:, 0:1]
            ).clamp(0.05, 2.0)
            anchor_map = torch.sigmoid(
                self.anchor_logit[k] + adaptive[:, 1:2]
            )
            delta = to_complex(delta_2ch)
            proposal = (
                z
                + step_map.squeeze(1) * delta
                + anchor_map.squeeze(1) * (z0 - z)
            )
            z = data_consistency(proposal, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class DualDomainRecurrentCNNAENetMC(nn.Module):
    """Recurrent AENet with image and k-space residual priors."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 6,
        kspace_depth: int = 4,
        coils: int = 4,
        anchor_init: float = 0.02,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.hidden = hidden
        self.image_prior = RecurrentResidualPrior(
            in_ch=4 + hidden,
            hidden=hidden,
            depth=depth,
            out_ch=2 + hidden,
            out_scale=0.1,
        )
        self.kspace_prior = KSpaceResidualPrior(
            coils=coils,
            hidden=hidden,
            depth=kspace_depth,
        )
        self.gate = nn.Conv2d(2 * hidden, hidden, kernel_size=1)
        self.log_step_image = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_step_kspace = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.anchor_logit = nn.Parameter(
            torch.full((steps,), math.log(anchor_init / (1.0 - anchor_init)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        h = torch.zeros(
            z.shape[0],
            self.hidden,
            z.shape[-2],
            z.shape[-1],
            device=z.device,
            dtype=z.real.dtype,
        )
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0), h), dim=1)
            out = self.image_prior(inp)
            delta_2ch = out[:, :2]
            h_new = out[:, 2:]
            gate = torch.sigmoid(self.gate(torch.cat((h, h_new), dim=1)))
            h = (1.0 - gate) * h + gate * torch.tanh(h_new)
            step_image = F.softplus(self.log_step_image[k]).clamp(0.05, 2.0)
            anchor = torch.sigmoid(self.anchor_logit[k])
            proposal = (
                z
                + step_image * to_complex(delta_2ch)
                + anchor * (z0 - z)
            )
            k_prop = multicoil_forward(proposal, smap)
            k_delta = channels_to_complex(
                self.kspace_prior(k_prop, y, self.mask)
            )
            step_kspace = F.softplus(self.log_step_kspace[k]).clamp(0.05, 2.0)
            k_new = y + (1.0 - self.mask[None, None]) * (
                k_prop + step_kspace * k_delta
            )
            z = multicoil_adjoint(k_new, smap)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class DualDomainVarNetSharedMC(nn.Module):
    """Parameter-matched shared Double-domain VarNet control."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 6,
        kspace_depth: int = 4,
        coils: int = 4,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.image_prior = RecurrentResidualPrior(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=0.1,
        )
        self.kspace_prior = KSpaceResidualPrior(
            coils=coils,
            hidden=hidden,
            depth=kspace_depth,
        )
        self.log_step_image = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_step_kspace = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta = to_complex(self.image_prior(inp))
            step_image = F.softplus(self.log_step_image[k]).clamp(0.05, 2.0)
            proposal = z + step_image * delta
            k_prop = multicoil_forward(proposal, smap)
            k_delta = channels_to_complex(
                self.kspace_prior(k_prop, y, self.mask)
            )
            step_kspace = F.softplus(self.log_step_kspace[k]).clamp(0.05, 2.0)
            k_new = y + (1.0 - self.mask[None, None]) * (
                k_prop + step_kspace * k_delta
            )
            z = multicoil_adjoint(k_new, smap)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class FeatureImageVarNetMC(nn.Module):
    """FI-VarNet-style shared baseline with feature-domain state."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 6,
        feat_ch: int = 32,
        attention: bool = False,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.feat_ch = feat_ch
        self.feat_proj = nn.Conv2d(2, feat_ch, kernel_size=1)
        self.inject = nn.Conv2d(2, feat_ch, kernel_size=1)
        self.prior = RecurrentResidualPrior(
            in_ch=feat_ch + 2,
            hidden=hidden,
            depth=depth,
            out_ch=feat_ch + 2,
            out_scale=0.1,
        )
        self.attn = (
            nn.Sequential(
                nn.Conv2d(feat_ch + 1, hidden, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            if attention
            else None
        )
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        feat = self.feat_proj(to_2ch(z0))
        trajectory = []
        for k in range(self.steps):
            attn = None
            if self.attn is not None:
                missing = (1.0 - self.mask[None, None]).to(feat.dtype)
                missing = missing.expand(feat.shape[0], -1, -1, -1)
                attn = self.attn(torch.cat((feat, missing), dim=1))
                feat_in = feat * (1.0 + attn)
            else:
                feat_in = feat
            out = self.prior(torch.cat((feat_in, to_2ch(z0)), dim=1))
            feat_new = out[:, : self.feat_ch]
            delta = to_complex(out[:, self.feat_ch :])
            if attn is not None:
                delta = delta * attn.squeeze(1)
                feat_new = feat_new * (1.0 + attn)
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            z = data_consistency(z + step * delta, y, smap, self.mask)
            feat = feat_new + self.inject(to_2ch(z))
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class VarNetSharedMC(nn.Module):
    """Weight-tied unrolled baseline with multicoil data consistency."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 8,
        hidden: int = 32,
        depth: int = 4,
        prior_factory=ComplexPrior,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = prior_factory(in_ch=4, hidden=hidden, depth=depth)
        self.log_step = nn.Parameter(torch.full((steps,), np.log(np.expm1(0.5))))

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta = to_complex(self.prior(inp))
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            z = data_consistency(z + step * delta, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class VarNetMC(nn.Module):
    """Non-shared unrolled baseline with the same parameter family."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 8,
        hidden: int = 32,
        depth: int = 4,
        prior_factory=ComplexPrior,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.priors = nn.ModuleList(
            [prior_factory(in_ch=4, hidden=hidden, depth=depth) for _ in range(steps)]
        )
        self.log_step = nn.Parameter(torch.full((steps,), np.log(np.expm1(0.5))))

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for k, prior in enumerate(self.priors):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta = to_complex(prior(inp))
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            z = data_consistency(z + step * delta, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class DirectUNetMC(nn.Module):
    """End-to-end image U-Net without an explicit data-consistency layer."""

    def __init__(self, mask: torch.Tensor, hidden: int = 32, depth: int = 3):
        super().__init__()
        del mask
        self.net = ComplexUNet(
            in_ch=2,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=1.0,
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        del y, smap
        delta = to_complex(self.net(to_2ch(z0)))
        output = z0 + delta
        return (output, []) if return_trajectory else output


class MoDLStyleMC(nn.Module):
    """MoDL-style unrolling with conjugate-gradient data consistency."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 6,
        hidden: int = 32,
        depth: int = 3,
        cg_steps: int = 3,
        rho: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.cg_steps = cg_steps
        self.rho = rho
        self.denoiser = ComplexUNet(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=0.2,
        )

    def _normal(self, x, smap):
        return multicoil_adjoint(
            self.mask[None, None] * multicoil_forward(x, smap),
            smap,
        ) + self.rho * x

    def _cg_dc(self, proposal, y, smap):
        rhs = multicoil_adjoint(y, smap) + self.rho * proposal
        x = proposal
        residual = rhs - self._normal(x, smap)
        direction = residual
        rs = torch.sum(torch.abs(residual) ** 2, dim=(1, 2), keepdim=True)
        for _ in range(self.cg_steps):
            normal_dir = self._normal(direction, smap)
            denom = torch.sum(
                (torch.conj(direction) * normal_dir).real,
                dim=(1, 2),
                keepdim=True,
            ).clamp_min(1e-12)
            alpha = rs / denom
            x = x + alpha * direction
            residual = residual - alpha * normal_dir
            rs_new = torch.sum(
                torch.abs(residual) ** 2, dim=(1, 2), keepdim=True
            )
            beta = rs_new / rs.clamp_min(1e-12)
            direction = residual + beta * direction
            rs = rs_new
        return x

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for _ in range(self.steps):
            delta = to_complex(
                self.denoiser(torch.cat((to_2ch(z), to_2ch(z0)), dim=1))
            )
            z = self._cg_dc(z + delta, y, smap)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class ISTASoftNetMC(nn.Module):
    """ISTA-style learned soft-thresholding network with hard DC."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        threshold_init: float = 0.01,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.transform = nn.Conv2d(2, hidden, 3, padding=1)
        self.inverse = nn.Conv2d(hidden, 2, 3, padding=1)
        self.threshold = nn.Parameter(torch.tensor(float(threshold_init)))
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            features = F.silu(self.transform(to_2ch(z)))
            threshold = F.softplus(self.threshold)
            scale = torch.clamp(
                1.0 - threshold / torch.abs(features).clamp_min(1e-8),
                min=0.0,
            )
            features = features * scale
            delta = to_complex(self.inverse(features))
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            z = data_consistency(z + step * delta, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


def complex_soft_threshold(z: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    magnitude = torch.abs(z)
    scale = torch.clamp(1.0 - threshold / magnitude.clamp_min(1e-8), min=0.0)
    return z * scale


class FISTAStyleMC(nn.Module):
    """FISTA-style momentum and learned soft-thresholding with hard DC."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 3,
        threshold_init: float = 0.01,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = ComplexUNet(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=0.2,
        )
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_threshold = nn.Parameter(
            torch.tensor([math.log(math.expm1(threshold_init))])
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        x = z0
        t = 1.0
        trajectory = []
        threshold = F.softplus(self.log_threshold[0])
        for k in range(self.steps):
            delta = to_complex(
                self.prior(torch.cat((to_2ch(x), to_2ch(z0)), dim=1))
            )
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            proposal = complex_soft_threshold(x + step * delta, threshold)
            v = data_consistency(proposal, y, smap, self.mask)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            x = v + ((t - 1.0) / t_new) * (v - z)
            z = v
            t = t_new
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class ADMMStyleMC(nn.Module):
    """ADMM-style splitting with learned denoising and hard DC."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 10,
        hidden: int = 32,
        depth: int = 3,
        threshold_init: float = 0.01,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = ComplexUNet(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=0.2,
        )
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_threshold = nn.Parameter(
            torch.tensor([math.log(math.expm1(threshold_init))])
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        z = z0
        u = z0
        dual = torch.zeros_like(z0)
        trajectory = []
        threshold = F.softplus(self.log_threshold[0])
        for k in range(self.steps):
            proposal = u - dual
            delta = to_complex(
                self.prior(torch.cat((to_2ch(proposal), to_2ch(z0)), dim=1))
            )
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            z = data_consistency(proposal + step * delta, y, smap, self.mask)
            u = complex_soft_threshold(z + dual, threshold)
            dual = dual + z - u
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class LPDNetMC(nn.Module):
    """Learned primal-dual style updates with a learned primal prior."""

    def __init__(
        self,
        mask: torch.Tensor,
        steps: int = 8,
        hidden: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = ComplexUNet(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=0.2,
        )
        self.log_step_primal = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_step_dual = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        primal = z0
        dual = torch.zeros_like(y)
        trajectory = []
        for k in range(self.steps):
            step_dual = F.softplus(self.log_step_dual[k]).clamp(0.05, 2.0)
            dual = dual + step_dual * self.mask[None, None] * (
                multicoil_forward(primal, smap) - y
            )
            step_primal = F.softplus(self.log_step_primal[k]).clamp(0.05, 2.0)
            delta = to_complex(
                self.prior(torch.cat((to_2ch(primal), to_2ch(z0)), dim=1))
            )
            primal = primal - step_primal * multicoil_adjoint(dual, smap) + delta
            primal = data_consistency(primal, y, smap, self.mask)
            if return_trajectory:
                trajectory.append(primal)
        return (primal, trajectory) if return_trajectory else primal


class AttentionUNetMC(nn.Module):
    """Direct U-Net with a learned channel-attention gate."""

    def __init__(self, mask: torch.Tensor, hidden: int = 32, depth: int = 3):
        super().__init__()
        del mask
        self.attention = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.net = ComplexUNet(
            in_ch=4,
            hidden=hidden,
            depth=depth,
            out_ch=2,
            out_scale=1.0,
        )

    def forward(self, z0, y, smap, return_trajectory=False):
        del y, smap
        image_channels = to_2ch(z0)
        gate = self.attention(image_channels)
        delta = to_complex(
            self.net(torch.cat((image_channels * (1.0 + gate), image_channels), dim=1))
        )
        output = z0 + delta
        return (output, []) if return_trajectory else output


class ZeroFilled(nn.Module):
    def forward(self, z0, y, smap, return_trajectory=False):
        return (z0, []) if return_trajectory else z0


class M4RawDataset(torch.utils.data.Dataset):
    def __init__(self, gt: torch.Tensor, smap: torch.Tensor, kspace: torch.Tensor, indices):
        self.gt = gt[indices]
        self.smap = smap[indices]
        self.kspace = kspace[indices]

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        return self.gt[i], self.smap[i], self.kspace[i]


def make_inputs(gt, smap, kspace, mask, device):
    gt = gt.to(device, non_blocking=True)
    smap = smap.to(device, non_blocking=True)
    kspace = kspace.to(device, non_blocking=True)
    mask = mask.to(device)
    y = kspace * mask[None, None]
    z0 = multicoil_adjoint(y, smap)
    return gt, y, z0


def make_loader(gt, smap, kspace, indices, batch_size, shuffle, seed):
    ds = M4RawDataset(gt, smap, kspace, indices)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader, mask, device):
    model.eval()
    vals = {k: [] for k in ("psnr", "ssim", "nmse")}
    for gt_batch, smap_batch, kspace_batch in loader:
        gt, y, z0 = make_inputs(gt_batch, smap_batch, kspace_batch, mask, device)
        recon = model(z0, y, smap_batch.to(device))
        m = metric_batch(recon, gt)
        for k in vals:
            vals[k].extend(m[k].detach().cpu().tolist())
    out = {}
    for k, v in vals.items():
        out[k] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def train_one(model, train_loader, val_loader, mask, device, args):
    model.to(device)
    ema_model = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.03
    )
    best_val = None
    best_state = None
    history = []
    accum = max(1, args.grad_accum)
    num_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (gt_batch, smap_batch, kspace_batch) in enumerate(
            train_loader, start=1
        ):
            if args.augment:
                theta = torch.rand(gt_batch.shape[0]) * (2.0 * np.pi)
                phasor = torch.polar(torch.ones_like(theta), theta)
                phasor = phasor.to(gt_batch.dtype)
                gt_batch = gt_batch * phasor[:, None, None]
                kspace_batch = kspace_batch * phasor[:, None, None, None]
            gt, y, z0 = make_inputs(gt_batch, smap_batch, kspace_batch, mask, device)
            recon, trajectory = model(
                z0, y, smap_batch.to(device), return_trajectory=True
            )
            loss = training_loss_fn(
                recon,
                gt,
                trajectory,
                args.deep_supervision,
                args.nmse_weight,
            ) / accum
            loss.backward()
            running += float(loss.item())
            if step % accum == 0 or step == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
        sched.step()
        val_model = evaluate(model, val_loader, mask, device)
        if ema_model is not None:
            ema_decay = min(args.ema_decay, (epoch + 1) / (epoch + 10))
            update_ema(ema_model, model, ema_decay)
            val_ema = evaluate(ema_model, val_loader, mask, device)
            if epoch + 1 >= args.ema_start_epoch:
                selected_name = "ema"
                val = val_ema
            else:
                selected_name = "model"
                val = val_model
        else:
            val_ema = None
            selected_name = "model"
            val = val_model
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, len(train_loader)),
            **val,
            "model_psnr": val_model["psnr"],
            "ema_psnr": None if val_ema is None else val_ema["psnr"],
            "selected": selected_name,
            "lr": sched.get_last_lr()[0],
        }
        history.append(row)
        if best_val is None or val["psnr"] > best_val["psnr"]:
            best_val = {"epoch": epoch + 1, "source": selected_name, **val}
            source_model = ema_model if selected_name == "ema" else model
            best_state = copy.deepcopy(source_model.state_dict())
        ema_text = "n/a" if val_ema is None else f"{val_ema['psnr']:.3f}"
        print(
            f"[{type(model).__name__}] epoch {epoch + 1}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val_psnr={val['psnr']:.3f} "
            f"val_ssim={val['ssim']:.4f} "
            f"model_psnr={val_model['psnr']:.3f} "
            f"ema_psnr={ema_text}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=r"F:\MRI data\V1.6\prepared\m4raw_smoke.pt",
    )
    ap.add_argument("--mask-key", default="r4_s42")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument(
        "--arch",
        choices=[
            "cnn",
            "unet",
            "recurrent",
            "recurrent_cnn",
            "adaptive_recurrent_cnn",
            "dual_domain_recurrent_cnn",
            "dual_domain_shared",
            "feature_image_varnet",
            "feature_image_attention",
            "official_varnet",
            "official_humusnet",
            "direct_unet",
            "modl_style",
            "ista_soft",
            "fista_style",
            "admm_style",
            "lpd_net",
            "attention_unet",
        ],
        default="cnn",
    )
    ap.add_argument("--anchor-init", type=float, default=0.8)
    ap.add_argument("--coils", type=int, default=4)
    ap.add_argument("--feature-channels", type=int, default=32)
    ap.add_argument("--smap-refine", action="store_true")
    ap.add_argument("--smap-hidden", type=int, default=32)
    ap.add_argument("--smap-depth", type=int, default=3)
    ap.add_argument("--official-varnet-cascades", type=int, default=12)
    ap.add_argument("--official-varnet-chans", type=int, default=18)
    ap.add_argument("--official-humus-cascades", type=int, default=8)
    ap.add_argument("--official-humus-embed", type=int, default=66)
    ap.add_argument("--official-humus-checkpoint", action="store_true")
    ap.add_argument("--modl-steps", type=int, default=6)
    ap.add_argument("--cg-steps", type=int, default=3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--deep-supervision", type=float, default=0.15)
    ap.add_argument(
        "--nmse-weight",
        type=float,
        default=0.0,
        help="optional extra magnitude NMSE term for PSNR-oriented training",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="enable EMA weights when greater than zero, e.g. 0.999",
    )
    ap.add_argument("--ema-start-epoch", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument(
        "--only",
        choices=["all", "aenet", "varnet", "varnet_shared", "baseline"],
        default="all",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="optional prefix tag for checkpoints/results")
    ap.add_argument("--device", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    gt = data["gt_complex"]
    smap = data["smap"]
    kspace = data["kspace"]
    masks = data["masks"]
    split = {k: list(v) for k, v in data["split"].items()}
    if args.max_train:
        split["train"] = split["train"][: args.max_train]
    if args.max_val:
        split["val"] = split["val"][: args.max_val]
    if args.max_test:
        split["test"] = split["test"][: args.max_test]
    mask = masks[args.mask_key].float()

    train_loader = make_loader(
        gt, smap, kspace, split["train"], args.batch_size, True, args.seed
    )
    val_loader = make_loader(
        gt, smap, kspace, split["val"], args.batch_size, False, args.seed
    )
    test_loader = make_loader(
        gt, smap, kspace, split["test"], args.batch_size, False, args.seed
    )
    dataset_name = data.get("config", {}).get("dataset", "m4raw")
    print(
        f"dataset={dataset_name} samples={len(gt)} train={len(split['train'])} "
        f"val={len(split['val'])} test={len(split['test'])} device={device}"
    )
    zero_filled = evaluate(ZeroFilled(), test_loader, mask, device)
    print("zero_filled=", zero_filled)

    model_defs = {}
    prior_factory = (
        ComplexUNet
        if args.arch in ("unet", "recurrent")
        else ComplexPrior
    )
    if args.only in ("all", "aenet"):
        if args.arch == "dual_domain_recurrent_cnn":
            model_defs["aenet"] = lambda: DualDomainRecurrentCNNAENetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                coils=args.coils,
                anchor_init=args.anchor_init,
            )
        elif args.arch == "adaptive_recurrent_cnn":
            model_defs["aenet"] = lambda: AdaptiveRecurrentCNNAENetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                anchor_init=args.anchor_init,
            )
        elif args.arch == "recurrent_cnn":
            model_defs["aenet"] = lambda: RecurrentCNNAENetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                anchor_init=args.anchor_init,
            )
        elif args.arch == "recurrent":
            model_defs["aenet"] = lambda: RecurrentAENetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                anchor_init=args.anchor_init,
            )
        else:
            model_defs["aenet"] = lambda: AENetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                prior_factory=prior_factory,
                anchor_init=args.anchor_init,
            )
    if args.only in ("all", "varnet_shared"):
        if args.arch == "dual_domain_shared":
            model_defs["varnet_shared"] = lambda: DualDomainVarNetSharedMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                coils=args.coils,
            )
        elif args.arch == "feature_image_varnet":
            model_defs["varnet_shared"] = lambda: FeatureImageVarNetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                feat_ch=args.feature_channels,
            )
        elif args.arch == "feature_image_attention":
            model_defs["varnet_shared"] = lambda: FeatureImageVarNetMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                feat_ch=args.feature_channels,
                attention=True,
            )
        elif args.arch == "official_varnet":
            model_defs["varnet_shared"] = lambda: OfficialVarNetAdapter(
                mask,
                num_cascades=args.official_varnet_cascades,
                chans=args.official_varnet_chans,
            )
        elif args.arch == "official_humusnet":
            model_defs["varnet_shared"] = lambda: OfficialHUMUSNetAdapter(
                mask,
                num_cascades=args.official_humus_cascades,
                embed_dim=args.official_humus_embed,
                use_checkpoint=args.official_humus_checkpoint,
            )
        else:
            model_defs["varnet_shared"] = lambda: VarNetSharedMC(
                mask,
                args.steps,
                args.hidden,
                args.depth,
                prior_factory=prior_factory,
            )
    if args.only in ("all", "varnet"):
        model_defs["varnet"] = lambda: VarNetMC(
            mask,
            args.steps,
            args.hidden,
            args.depth,
            prior_factory=prior_factory,
        )
    if args.only in ("all", "baseline"):
        if args.arch == "direct_unet":
            model_defs["baseline"] = lambda: DirectUNetMC(
                mask,
                hidden=args.hidden,
                depth=args.depth,
            )
        elif args.arch == "modl_style":
            model_defs["baseline"] = lambda: MoDLStyleMC(
                mask,
                steps=args.modl_steps,
                hidden=args.hidden,
                depth=args.depth,
                cg_steps=args.cg_steps,
            )
        elif args.arch == "ista_soft":
            model_defs["baseline"] = lambda: ISTASoftNetMC(
                mask,
                steps=args.steps,
                hidden=args.hidden,
            )
        elif args.arch == "fista_style":
            model_defs["baseline"] = lambda: FISTAStyleMC(
                mask,
                steps=args.steps,
                hidden=args.hidden,
                depth=args.depth,
            )
        elif args.arch == "admm_style":
            model_defs["baseline"] = lambda: ADMMStyleMC(
                mask,
                steps=args.steps,
                hidden=args.hidden,
                depth=args.depth,
            )
        elif args.arch == "lpd_net":
            model_defs["baseline"] = lambda: LPDNetMC(
                mask,
                steps=args.steps,
                hidden=args.hidden,
                depth=args.depth,
            )
        elif args.arch == "attention_unet":
            model_defs["baseline"] = lambda: AttentionUNetMC(
                mask,
                hidden=args.hidden,
                depth=args.depth,
            )
    if args.smap_refine:
        refined_defs = {}
        for name, factory in model_defs.items():
            refined_defs[name] = (
                lambda base_factory=factory: SmapRefinedModel(
                    base_factory(),
                    coils=args.coils,
                    hidden=args.smap_hidden,
                    depth=args.smap_depth,
                )
            )
        model_defs = refined_defs

    results = {
        "dataset": dataset_name,
        "zero_filled": zero_filled,
        "config": vars(args),
        "device": str(device),
        "models": {},
    }
    tag = f"_{args.tag}" if args.tag else ""
    if dataset_name.startswith("fastmri") and args.tag.startswith("fastmri_knee_"):
        tag = f"_{args.tag[len('fastmri_knee_'):]}"
    checkpoint_prefix = (
        "fastmri_knee" if dataset_name.startswith("fastmri") else "m4raw"
    )
    for name, factory in model_defs.items():
        model = factory()
        params = count_parameters(model)
        t0 = time.time()
        history, best_val = train_one(
            model, train_loader, val_loader, mask, device, args
        )
        test = evaluate(model, test_loader, mask, device)
        ckpt = ROOT / (
            f"{checkpoint_prefix}{tag}_{args.mask_key}_{name}_s{args.seed}.pt"
        )
        torch.save(
            {"model": model.state_dict(), "config": vars(args), "mask": mask.cpu()},
            ckpt,
        )
        results["models"][name] = {
            "params": params,
            "checkpoint": str(ckpt),
            "train_sec": round(time.time() - t0, 2),
            "best_val": best_val,
            "test": test,
            "history": history,
        }
        print(f"{name}: params={params} test={test}")

    out = (
        Path(args.output)
        if args.output
        else ROOT
        / f"m4raw{tag}_results_{args.mask_key}_{args.only}_s{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
