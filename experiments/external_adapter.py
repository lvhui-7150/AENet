"""Adapters that run official fastMRI VarNet and HUMUS-Net on M4Raw tensors."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def add_external_paths() -> None:
    parents = Path(__file__).resolve().parents
    roots = [parents[1] / "external", parents[2] / "external"]
    for root in roots:
        for name in ("fastMRI", "HUMUS-Net"):
            path = root / name
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))


def mask_to_fastmri(mask: torch.Tensor) -> torch.Tensor:
    # Official fastMRI expects (B, 1, 1, W, 1).
    if mask.ndim == 2:
        line_mask = mask[0]
    elif mask.ndim == 4:
        line_mask = mask[0, 0, 0]
    else:
        raise ValueError(f"unsupported mask shape: {tuple(mask.shape)}")
    return line_mask.view(1, 1, 1, -1, 1).bool()


def install_timm_layer_compat() -> None:
    """Expose modern timm layers under the old import path used by HUMUS-Net."""
    try:
        import timm.models.layers  # noqa: F401
        return
    except Exception:
        pass
    import timm.layers as timm_layers

    module = types.ModuleType("timm.models.layers")
    for name in ("DropPath", "to_2tuple", "trunc_normal_"):
        setattr(module, name, getattr(timm_layers, name))
    sys.modules["timm.models.layers"] = module


class OfficialVarNetAdapter(nn.Module):
    """Official fastMRI VarNet with an M4Raw tensor interface."""

    def __init__(
        self,
        mask: torch.Tensor,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        super().__init__()
        add_external_paths()
        from fastmri.models import VarNet

        self.model = VarNet(
            num_cascades=num_cascades,
            sens_chans=sens_chans,
            sens_pools=sens_pools,
            chans=chans,
            pools=pools,
            mask_center=mask_center,
        )
        self.register_buffer("mask5", mask_to_fastmri(mask))

    def forward(self, z0, y, smap, return_trajectory=False):
        del z0, smap
        masked_kspace = torch.view_as_real(y)
        output = self.model(masked_kspace, self.mask5.to(y.device))
        output = torch.complex(output, torch.zeros_like(output))
        return (output, []) if return_trajectory else output


class OfficialHUMUSNetAdapter(nn.Module):
    """Official HUMUS-Net with an M4Raw tensor interface."""

    def __init__(
        self,
        mask: torch.Tensor,
        num_cascades: int = 8,
        sens_chans: int = 16,
        sens_pools: int = 4,
        embed_dim: int = 66,
        depths=(2, 2, 2),
        num_heads=(3, 6, 12),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        resi_connection: str = "1conv",
        bottleneck_depth: int = 2,
        bottleneck_heads: int = 24,
        conv_downsample_first: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        add_external_paths()
        install_timm_layer_compat()
        from models.humus_net import HUMUSNet

        img_size = list(mask.shape[-2:])
        self.model = HUMUSNet(
            num_cascades=num_cascades,
            sens_chans=sens_chans,
            sens_pools=sens_pools,
            mask_center=True,
            num_adj_slices=1,
            img_size=img_size,
            patch_size=1,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            use_checkpoint=use_checkpoint,
            resi_connection=resi_connection,
            bottleneck_depth=bottleneck_depth,
            bottleneck_heads=bottleneck_heads,
            conv_downsample_first=conv_downsample_first,
        )
        self.register_buffer("mask5", mask_to_fastmri(mask))

    def forward(self, z0, y, smap, return_trajectory=False):
        del z0, smap
        masked_kspace = torch.view_as_real(y)
        output = self.model(masked_kspace, self.mask5.to(y.device))
        output = torch.complex(output, torch.zeros_like(output))
        return (output, []) if return_trajectory else output
