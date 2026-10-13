# -*- coding: utf-8 -*-
"""TPAMI-oriented AENet/VarNet experiment pipeline.

This script is intentionally independent from the original smoke-test code.
It uses the prepared datasets already referenced by the repository, keeps the
sampled k-space coefficients fixed at every iteration, trains with deep
supervision, and evaluates the best validation checkpoint with one metric
definition for all models.

The AENet update is

    z_{k+1} = A^H(y) + P_0(z_k + D_theta(z_k, A^H y)),

which is the practical single-coil Cartesian realization of a nullspace-aware
anchored update: the learned residual can only change unmeasured frequencies.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent

# Set AENET_DATA_DIR to the directory holding the prepared datasets, e.g.
# /path/to/aenet/data. Without it the code falls back to data/ inside this
# repository.
_DATA_DIR = Path(os.environ.get("AENET_DATA_DIR", ""))
if _DATA_DIR:
    FASTMRI_PREPARED = str(_DATA_DIR / "fastmri_128_prepared.pt")
    IXI_PREPARED = str(_DATA_DIR / "complex_ixi_data_v1.pt")
    FMRI320_META = str(_DATA_DIR / "fastmri_320_meta.pt")
    FMRI320_CHUNK_TEMPLATE = str(_DATA_DIR / "fastmri_320_gt_chunk_{:03d}.pt")
    FASTMRI_H5_DIR = str(_DATA_DIR / "knee_singlecoil_val" / "singlecoil_val")
else:
    _LOCAL_DATA_DIR = ROOT.parent / "data"
    FASTMRI_PREPARED = str(_LOCAL_DATA_DIR / "fastmri_128_prepared.pt")
    IXI_PREPARED = str(_LOCAL_DATA_DIR / "complex_ixi_data_v1.pt")
    FMRI320_META = str(_LOCAL_DATA_DIR / "fastmri_320_meta.pt")
    FMRI320_CHUNK_TEMPLATE = str(_LOCAL_DATA_DIR / "fastmri_320_gt_chunk_{:03d}.pt")
    FASTMRI_H5_DIR = str(_LOCAL_DATA_DIR / "knee_singlecoil_val" / "singlecoil_val")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho")


def to_2ch(z: torch.Tensor) -> torch.Tensor:
    return torch.stack((z.real, z.imag), dim=1)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    # x/y are Bx1xHxW; data_range is Bx1x1x1.
    win = gaussian_window(7, 1.5, x.device, x.dtype)
    mu_x = F.conv2d(x, win, padding=3)
    mu_y = F.conv2d(y, win, padding=3)
    var_x = F.conv2d(x * x, win, padding=3) - mu_x * mu_x
    var_y = F.conv2d(y * y, win, padding=3) - mu_y * mu_y
    cov = F.conv2d(x * y, win, padding=3) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    )
    return score.flatten(1).mean(1)


def metric_batch(recon: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    xr = recon.abs().unsqueeze(1)
    yg = gt.abs().unsqueeze(1)
    peak = yg.flatten(1).amax(1).clamp_min(1e-6).view(-1, 1, 1, 1)
    mse = (xr - yg).square().flatten(1).mean(1).clamp_min(1e-12)
    psnr_v = 10 * torch.log10(peak.flatten() ** 2 / mse)
    ssim_v = ssim(xr, yg, peak)
    nmse_v = (xr - yg).square().flatten(1).sum(1) / yg.square().flatten(1).sum(1).clamp_min(1e-12)
    return {"psnr": psnr_v, "ssim": ssim_v, "nmse": nmse_v}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class ComplexPrior(nn.Module):
    """Residual U-Net prior with a bounded initial output scale."""

    def __init__(self, in_ch: int = 4, hidden: int = 32, depth: int = 6):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(depth)])
        self.mid = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, 3, padding=1),
            nn.GroupNorm(8 if hidden * 2 % 8 == 0 else 1, hidden * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = self.blocks(h)
        h = self.mid(h)
        return 0.2 * self.out(h)


class AENetTPAMI(nn.Module):
    """Shared anchored network; only the nullspace is modified per step."""

    def __init__(self, mask: torch.Tensor, steps: int = 12, hidden: int = 32, depth: int = 6):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = ComplexPrior(in_ch=4, hidden=hidden, depth=depth)
        # Step-wise FiLM keeps the operator weight-tied while allowing each
        # equilibrium iteration to specialize its effective update.
        self.step_scale = nn.Parameter(torch.ones(steps, 2))
        self.step_bias = nn.Parameter(torch.zeros(steps, 2))
        self.log_step = nn.Parameter(torch.full((steps,), math.log(math.expm1(0.5))))
        self.log_anchor = nn.Parameter(torch.full((steps,), math.log(math.expm1(0.8))))

    def forward(self, z0: torch.Tensor, y: torch.Tensor, return_trajectory: bool = False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta_2ch = self.prior(inp)
            delta_2ch = delta_2ch * self.step_scale[k].view(1, 2, 1, 1) + self.step_bias[k].view(1, 2, 1, 1)
            delta = to_complex(delta_2ch)
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            anchor = F.softplus(self.log_anchor[k]).clamp(0.05, 2.0)
            proposal = z + step * delta + anchor * (z0 - z)
            proposal_k = fft2c(proposal)
            # The prepared data are noiseless retrospective measurements, so
            # exact data consistency is the appropriate default. The network
            # learns only the missing Fourier coefficients.
            z = ifft2c(self.mask * y + (1 - self.mask) * proposal_k)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class VarNetSharedComparable(nn.Module):
    """Weight-tied reference with the same parameter family as AENet."""

    def __init__(self, mask: torch.Tensor, steps: int = 12, hidden: int = 32, depth: int = 6):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.prior = ComplexPrior(in_ch=4, hidden=hidden, depth=depth)
        self.log_step = nn.Parameter(torch.full((steps,), math.log(math.expm1(0.5))))
        self.log_dc = nn.Parameter(torch.full((steps,), math.log(math.expm1(0.8))))

    def forward(self, z0: torch.Tensor, y: torch.Tensor, return_trajectory: bool = False):
        z = z0
        trajectory = []
        for k in range(self.steps):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta = to_complex(self.prior(inp))
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            dc = F.softplus(self.log_dc[k]).clamp(0.05, 2.0)
            proposal = z + step * delta
            z = proposal - dc * ifft2c(self.mask * (fft2c(proposal) - y))
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class VarNetComparable(nn.Module):
    """Non-shared baseline with the same prior family and hard data consistency."""

    def __init__(self, mask: torch.Tensor, steps: int = 12, hidden: int = 32, depth: int = 6):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.steps = steps
        self.priors = nn.ModuleList([ComplexPrior(in_ch=4, hidden=hidden, depth=depth) for _ in range(steps)])
        self.log_step = nn.Parameter(torch.full((steps,), math.log(math.expm1(0.5))))

    def forward(self, z0: torch.Tensor, y: torch.Tensor, return_trajectory: bool = False):
        z = z0
        trajectory = []
        for k, prior in enumerate(self.priors):
            inp = torch.cat((to_2ch(z), to_2ch(z0)), dim=1)
            delta = to_complex(prior(inp))
            step = F.softplus(self.log_step[k]).clamp(0.05, 2.0)
            proposal_k = fft2c(z + step * delta)
            z = ifft2c(self.mask * y + (1 - self.mask) * proposal_k)
            if return_trajectory:
                trajectory.append(z)
        return (z, trajectory) if return_trajectory else z


class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, gt: torch.Tensor, indices):
        self.gt = gt[indices]

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, i):
        return self.gt[i]


def load_data(dataset: str):
    if dataset == "fastmri":
        data = torch.load(FASTMRI_PREPARED, map_location="cpu", weights_only=False)
        return data["gt_complex"], data["masks"], data["split"]
    if dataset == "ixi":
        data = torch.load(IXI_PREPARED, map_location="cpu", weights_only=False)
        n = len(data["gt_complex"])
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(n, generator=g).tolist()
        return data["gt_complex"], {"default": data["mask"]}, {
            "train": idx[: int(0.8 * n)], "val": idx[int(0.8 * n): int(0.9 * n)], "test": idx[int(0.9 * n):]
        }
    if dataset == "fastmri_h5":
        files = sorted(glob.glob(os.path.join(FASTMRI_H5_DIR, "*.h5")))
        gt_parts, sizes = [], []
        for p in files:
            with h5py.File(p, "r") as f:
                img = torch.from_numpy(f["reconstruction_esc"][:]).float()
                img = img / img.amax().clamp_min(1e-6)
            gt_parts.append(torch.complex(img, torch.zeros_like(img)))
            sizes.append(len(img))
        gt = torch.cat(gt_parts)
        meta = torch.load(FMRI320_META, map_location="cpu", weights_only=False)
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        split = {}
        for name in ("train", "val", "test"):
            split[name] = [i for fi in meta["split"][f"{name}_files"] for i in range(offsets[fi], offsets[fi + 1])]
        return gt, {"r4_s42": meta["masks"]["r4_s42"]["mask"]}, split
    if dataset == "fastmri320full":
        parts = [torch.load(FMRI320_CHUNK_TEMPLATE.format(i), map_location="cpu", weights_only=False)["gt"] for i in range(8)]
        gt = torch.cat(parts)
        meta = torch.load(FMRI320_META, map_location="cpu", weights_only=False)
        file_of_slice = meta["slice_files"].tolist()
        by_file = {}
        for gi, fi in enumerate(file_of_slice):
            by_file.setdefault(fi, []).append(gi)
        split = {name: [gi for fi in meta["split"][f"{name}_files"] for gi in by_file[fi]] for name in ("train", "val", "test")}
        return gt, {"r4_s42": meta["masks"]["r4_s42"]["mask"]}, split
    raise ValueError(dataset)


def make_loader(gt, indices, batch_size, shuffle, seed):
    ds = MRIDataset(gt, indices)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0, pin_memory=torch.cuda.is_available())


def make_inputs(gt: torch.Tensor, mask: torch.Tensor, device: torch.device):
    gt = gt.to(device, non_blocking=True)
    mask = mask.to(device)
    y = fft2c(gt) * mask
    z0 = ifft2c(y)
    return gt, y, z0


def loss_fn(recon: torch.Tensor, gt: torch.Tensor, trajectory=None, deep_supervision: float = 0.15):
    final = metric_batch(recon, gt)
    loss = (1 - final["ssim"]).mean() + 0.5 * (recon.abs() - gt.abs()).abs().mean()
    if trajectory:
        weights = torch.linspace(deep_supervision, 1.0, len(trajectory), device=recon.device)
        weights = weights / weights.sum()
        aux = 0.0
        for w, z in zip(weights[:-1], trajectory[:-1]):
            # Avoid repeated windowed SSIM/NMSE evaluations on every intermediate
            # state; the final state still receives the full perceptual objective.
            aux = aux + w * (z.abs() - gt.abs()).abs().mean()
        loss = loss + deep_supervision * aux
    return loss


@torch.no_grad()
def evaluate(model, loader, mask, device):
    model.eval()
    vals = {k: [] for k in ("psnr", "ssim", "nmse")}
    for gt_batch in loader:
        gt, y, z0 = make_inputs(gt_batch, mask, device)
        recon = model(z0, y)
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.03)
    best_val = None
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for gt_batch in train_loader:
            if args.augment:
                if torch.rand(()) < 0.5:
                    gt_batch = torch.flip(gt_batch, (-1,))
                if torch.rand(()) < 0.5:
                    gt_batch = torch.flip(gt_batch, (-2,))
            gt, y, z0 = make_inputs(gt_batch, mask, device)
            opt.zero_grad(set_to_none=True)
            recon, trajectory = model(z0, y, return_trajectory=True)
            loss = loss_fn(recon, gt, trajectory, args.deep_supervision)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.item())
        sched.step()
        val = evaluate(model, val_loader, mask, device)
        row = {"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), **val, "lr": sched.get_last_lr()[0]}
        history.append(row)
        if best_val is None or val["psnr"] > best_val["psnr"]:
            best_val = {"epoch": epoch + 1, **val}
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{type(model).__name__}] epoch {epoch + 1}/{args.epochs} loss={row['train_loss']:.4f} val_psnr={val['psnr']:.3f} val_ssim={val['ssim']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fastmri", "fastmri_h5", "fastmri320full", "ixi"], default="fastmri")
    ap.add_argument("--mask-key", default="r4_s42")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--deep-supervision", type=float, default=0.15)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--only", choices=["all", "aenet", "varnet", "varnet_shared"], default="all")
    ap.add_argument("--max-train", type=int, default=0, help="optional cap for fast smoke tests")
    ap.add_argument("--max-val", type=int, default=0, help="optional cap for fast smoke tests")
    ap.add_argument("--max-test", type=int, default=0, help="optional cap for fast smoke tests")
    ap.add_argument("--output", default="", help="explicit JSON output path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gt, masks, split = load_data(args.dataset)
    if args.max_train:
        split["train"] = split["train"][: args.max_train]
    if args.max_val:
        split["val"] = split["val"][: args.max_val]
    if args.max_test:
        split["test"] = split["test"][: args.max_test]
    mask = masks.get(args.mask_key, masks.get("default")).float()
    train_loader = make_loader(gt, split["train"], args.batch_size, True, args.seed)
    val_loader = make_loader(gt, split["val"], args.batch_size, False, args.seed)
    test_loader = make_loader(gt, split["test"], args.batch_size, False, args.seed)
    print(f"dataset={args.dataset} train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} device={device}")
    print("zero_filled=", evaluate(ZeroFilled(), test_loader, mask, device))

    model_defs = {}
    if args.only in ("all", "aenet"):
        model_defs["aenet"] = lambda: AENetTPAMI(mask, args.steps, args.hidden, args.depth)
    if args.only in ("all", "varnet_shared"):
        model_defs["varnet_shared"] = lambda: VarNetSharedComparable(mask, args.steps, args.hidden, args.depth)
    if args.only in ("all", "varnet"):
        model_defs["varnet"] = lambda: VarNetComparable(mask, args.steps, args.hidden, args.depth)
    results = {"config": vars(args), "device": str(device), "models": {}}
    for name, factory in model_defs.items():
        model = factory()
        params = count_parameters(model)
        t0 = time.time()
        history, best_val = train_one(model, train_loader, val_loader, mask, device, args)
        test = evaluate(model, test_loader, mask, device)
        ckpt = ROOT / f"tpami_{args.dataset}_{name}_s{args.seed}.pt"
        torch.save({"model": model.state_dict(), "config": vars(args), "mask": mask.cpu()}, ckpt)
        results["models"][name] = {"params": params, "checkpoint": str(ckpt), "train_sec": round(time.time() - t0, 2), "best_val": best_val, "test": test, "history": history}
        print(f"{name}: params={params} test={test}")
    out = Path(args.output) if args.output else ROOT / f"tpami_results_{args.dataset}_{args.only}_s{args.seed}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {out}")


class ZeroFilled(nn.Module):
    def forward(self, z0, y):
        return z0


if __name__ == "__main__":
    main()
