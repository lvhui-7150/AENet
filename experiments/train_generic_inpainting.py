"""Train generic-image reconstruction models on BSD400/BSD68 inpainting.

The experiment uses a fixed-size RGB protocol. BSD400 is split into train and
validation images; the original BSD68 images are used for testing. The main
comparison contains a direct U-Net, a weight-tied unrolled U-Net, a non-shared
unrolled U-Net, and a recurrent AENet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    values = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    values = values / values.sum()
    return (values[:, None] * values[None, :])[None, None]


def image_metrics(recon: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    recon = recon.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (recon - target).square().flatten(1).mean(1).clamp_min(1e-12)
    psnr = 10.0 * torch.log10(1.0 / mse)
    nmse = (recon - target).square().flatten(1).sum(1) / (
        target.square().flatten(1).sum(1).clamp_min(1e-12)
    )

    window = gaussian_window(7, 1.5, recon.device, recon.dtype)
    ssim_values = []
    for channel in range(recon.shape[1]):
        x = recon[:, channel : channel + 1]
        y = target[:, channel : channel + 1]
        mu_x = F.conv2d(x, window, padding=3)
        mu_y = F.conv2d(y, window, padding=3)
        var_x = F.conv2d(x * x, window, padding=3) - mu_x.square()
        var_y = F.conv2d(y * y, window, padding=3) - mu_y.square()
        cov = F.conv2d(x * y, window, padding=3) - mu_x * mu_y
        c1 = 0.01**2
        c2 = 0.03**2
        score = ((2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)) / (
            (mu_x.square() + mu_y.square() + c1)
            * (var_x + var_y + c2)
        )
        ssim_values.append(score.flatten(1).mean(1))
    ssim = torch.stack(ssim_values, dim=1).mean(dim=1)
    return {"psnr": psnr, "ssim": ssim, "nmse": nmse}


class BSDInpaintingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        paths: list[Path],
        image_size: int,
        missing_rate: float,
        mask_seed: int,
        training: bool,
        augment: bool,
    ) -> None:
        self.root = root
        self.paths = paths
        self.image_size = image_size
        self.missing_rate = missing_rate
        self.mask_seed = mask_seed
        self.training = training
        self.augment = augment
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = self.image_size / min(width, height)
        new_size = (max(self.image_size, round(width * scale)),
                    max(self.image_size, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        width, height = image.size
        left = (width - self.image_size) // 2
        top = (height - self.image_size) // 2
        image = image.crop(
            (left, top, left + self.image_size, top + self.image_size)
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))

    def _fixed_mask_seed(self, path: Path) -> int:
        digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:8]
        return self.mask_seed + int(digest, 16)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        image = self._load_image(path)

        if self.training and self.augment:
            rng = np.random.default_rng(self.mask_seed + self.epoch * 100003 + index)
            if rng.random() < 0.5:
                image = image[:, :, ::-1].copy()

        mask_seed = (
            self.mask_seed + self.epoch * 100003 + index
            if self.training
            else self._fixed_mask_seed(path)
        )
        rng = np.random.default_rng(mask_seed)
        mask = (
            rng.random((1, self.image_size, self.image_size), dtype=np.float32)
            >= self.missing_rate
        ).astype(np.float32)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        mask_tensor = torch.from_numpy(mask)
        observed = image_tensor * mask_tensor
        return {
            "image": image_tensor,
            "observed": observed,
            "mask": mask_tensor,
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
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


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int,
        depth: int,
        out_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.out_scale = out_scale
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden),
        )
        self.downs = nn.ModuleList()
        channels = [hidden]
        current = hidden
        for _ in range(depth):
            next_channels = current * 2
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(current, next_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(
                        8 if next_channels % 8 == 0 else 1,
                        next_channels,
                    ),
                    nn.SiLU(inplace=True),
                    ResidualBlock(next_channels),
                )
            )
            current = next_channels
            channels.append(current)

        self.ups = nn.ModuleList()
        for index in range(depth):
            skip_channels = channels[-2 - index]
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(current, skip_channels, 1),
                        nn.Sequential(
                            nn.Conv2d(skip_channels * 2, skip_channels, 3, padding=1),
                            nn.GroupNorm(
                                8 if skip_channels % 8 == 0 else 1,
                                skip_channels,
                            ),
                            nn.SiLU(inplace=True),
                            ResidualBlock(skip_channels),
                        ),
                    ]
                )
            )
            current = skip_channels
        self.out = nn.Conv2d(current, out_channels, 3, padding=1)
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


def data_consistency(
    proposal: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return observed + (1.0 - mask) * proposal


class DirectUNet(nn.Module):
    def __init__(self, hidden: int, depth: int) -> None:
        super().__init__()
        self.net = TinyUNet(4, 3, hidden, depth, out_scale=1.0)

    def forward(self, observed, mask, return_trajectory=False):
        output = observed + self.net(torch.cat((observed, mask), dim=1))
        return (output, []) if return_trajectory else output


class WeightTiedUnrolled(nn.Module):
    def __init__(self, hidden: int, depth: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.prior = TinyUNet(7, 3, hidden, depth)
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, observed, mask, return_trajectory=False):
        state = observed
        trajectory = []
        for index in range(self.steps):
            delta = self.prior(torch.cat((state, observed, mask), dim=1))
            step = F.softplus(self.log_step[index]).clamp(0.05, 2.0)
            state = data_consistency(state + step * delta, observed, mask)
            if return_trajectory:
                trajectory.append(state)
        return (state, trajectory) if return_trajectory else state


class NonSharedUnrolled(nn.Module):
    def __init__(self, hidden: int, depth: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.priors = nn.ModuleList(
            [TinyUNet(7, 3, hidden, depth) for _ in range(steps)]
        )
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, observed, mask, return_trajectory=False):
        state = observed
        trajectory = []
        for index, prior in enumerate(self.priors):
            delta = prior(torch.cat((state, observed, mask), dim=1))
            step = F.softplus(self.log_step[index]).clamp(0.05, 2.0)
            state = data_consistency(state + step * delta, observed, mask)
            if return_trajectory:
                trajectory.append(state)
        return (state, trajectory) if return_trajectory else state


class RecurrentAENet(nn.Module):
    def __init__(self, hidden: int, depth: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.hidden = hidden
        self.prior = TinyUNet(7 + hidden, 3 + hidden, hidden, depth)
        self.gate = nn.Conv2d(2 * hidden, hidden, 1)
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )

    def forward(self, observed, mask, return_trajectory=False):
        state = observed
        recurrent = torch.zeros(
            state.shape[0],
            self.hidden,
            state.shape[-2],
            state.shape[-1],
            device=state.device,
            dtype=state.dtype,
        )
        trajectory = []
        for index in range(self.steps):
            output = self.prior(
                torch.cat((state, observed, mask, recurrent), dim=1)
            )
            delta = output[:, :3]
            recurrent_new = output[:, 3:]
            gate = torch.sigmoid(
                self.gate(torch.cat((recurrent, recurrent_new), dim=1))
            )
            recurrent = (1.0 - gate) * recurrent + gate * torch.tanh(recurrent_new)
            step = F.softplus(self.log_step[index]).clamp(0.05, 2.0)
            state = data_consistency(state + step * delta, observed, mask)
            if return_trajectory:
                trajectory.append(state)
        return (state, trajectory) if return_trajectory else state


class RecurrentAENetV2(nn.Module):
    """Wider recurrent AENet with learned anchor and per-channel step scaling."""

    def __init__(self, hidden: int, depth: int, steps: int,
                 anchor_init: float = 0.05) -> None:
        super().__init__()
        self.steps = steps
        self.hidden = hidden
        self.prior = TinyUNet(7 + hidden, 3 + hidden, hidden, depth)
        self.gate = nn.Conv2d(2 * hidden, hidden, 1)
        self.step_scale = nn.Parameter(torch.ones(steps, 3))
        self.step_bias = nn.Parameter(torch.zeros(steps, 3))
        self.log_step = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(0.5)))
        )
        self.log_anchor = nn.Parameter(
            torch.full((steps,), math.log(math.expm1(anchor_init)))
        )

    def forward(self, observed, mask, return_trajectory=False):
        state = observed
        recurrent = torch.zeros(
            state.shape[0],
            self.hidden,
            state.shape[-2],
            state.shape[-1],
            device=state.device,
            dtype=state.dtype,
        )
        trajectory = []
        for index in range(self.steps):
            output = self.prior(
                torch.cat((state, observed, mask, recurrent), dim=1)
            )
            delta = output[:, :3]
            recurrent_new = output[:, 3:]
            gate = torch.sigmoid(
                self.gate(torch.cat((recurrent, recurrent_new), dim=1))
            )
            recurrent = (1.0 - gate) * recurrent + gate * torch.tanh(recurrent_new)
            delta = (
                delta * self.step_scale[index].view(1, 3, 1, 1)
                + self.step_bias[index].view(1, 3, 1, 1)
            )
            step = F.softplus(self.log_step[index]).clamp(0.05, 2.0)
            anchor = F.softplus(self.log_anchor[index]).clamp(0.0, 1.0)
            proposal = state + step * delta + anchor * (observed - state)
            state = data_consistency(proposal, observed, mask)
            if return_trajectory:
                trajectory.append(state)
        return (state, trajectory) if return_trajectory else state


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    accumulated = {"psnr": [], "ssim": [], "nmse": []}
    for batch in loader:
        image = batch["image"].to(device)
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        recon = model(observed, mask)
        metrics = image_metrics(recon, image)
        for key in accumulated:
            accumulated[key].extend(metrics[key].cpu().tolist())
    result = {}
    for key, values in accumulated.items():
        result[key] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1))
    return result


def train_one(model, train_loader, val_loader, device, args):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.03,
    )
    best_val = None
    best_state = None
    history = []
    for epoch in range(args.epochs):
        train_loader.dataset.epoch = epoch
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            if args.deep_supervision > 0.0:
                recon, trajectory = model(
                    observed, mask, return_trajectory=True
                )
                targets = trajectory if trajectory else [recon]
                weights = np.linspace(
                    1.0 - args.deep_supervision,
                    1.0,
                    len(targets),
                    dtype=np.float32,
                )
                losses = [
                    F.l1_loss(candidate, image)
                    + args.mse_weight * F.mse_loss(candidate, image)
                    for candidate in targets
                ]
                loss = sum(
                    float(weight) * value
                    for weight, value in zip(weights, losses)
                ) / float(weights.sum())
            else:
                recon = model(observed, mask)
                loss = F.l1_loss(recon, image) + args.mse_weight * F.mse_loss(
                    recon, image
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += float(loss.item())
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            **validation,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if best_val is None or validation["psnr"] > best_val["psnr"]:
            best_val = {"epoch": epoch + 1, **validation}
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{type(model).__name__}] epoch {epoch + 1}/{args.epochs} "
            f"loss={row['train_loss']:.4f} val_psnr={validation['psnr']:.3f} "
            f"val_ssim={validation['ssim']:.4f}",
            flush=True,
        )
        if args.max_epochs and epoch + 1 >= args.max_epochs:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/path/to/aenet/data/inpainting/raw/denoising-datasets-main",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=180)
    parser.add_argument("--missing-rate", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--deep-supervision", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-count", type=int, default=350)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["direct_unet", "weight_tied", "non_shared", "aenet"],
        choices=["direct_unet", "weight_tied", "non_shared", "aenet", "aenet_v2"],
    )
    parser.add_argument(
        "--output",
        default="experiments/generic_inpainting_bsd_results.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    train_root = root / "BSD400"
    test_root = root / "BSD68" / "original"
    train_paths = sorted(train_root.glob("*.png"))
    test_paths = sorted(test_root.glob("*.png"))
    if len(train_paths) < args.train_count + args.val_count:
        raise RuntimeError("not enough BSD400 images for the requested split")
    if not test_paths:
        raise RuntimeError(f"no BSD68 images found under {test_root}")

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(train_paths))
    selected_paths = [
        train_paths[index]
        for index in permutation[: args.train_count + args.val_count]
    ]
    train_paths = selected_paths[: args.train_count]
    val_paths = selected_paths[args.train_count :]
    if args.max_train:
        train_paths = train_paths[: args.max_train]
    if args.max_val:
        val_paths = val_paths[: args.max_val]
    if args.max_test:
        test_paths = test_paths[: args.max_test]

    train_dataset = BSDInpaintingDataset(
        train_root,
        train_paths,
        args.image_size,
        args.missing_rate,
        args.seed,
        True,
        args.augment,
    )
    val_dataset = BSDInpaintingDataset(
        train_root,
        val_paths,
        args.image_size,
        args.missing_rate,
        args.seed + 1000003,
        False,
        False,
    )
    test_dataset = BSDInpaintingDataset(
        test_root,
        test_paths,
        args.image_size,
        args.missing_rate,
        args.seed + 2000003,
        False,
        False,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)

    print(
        f"dataset=BSD400/BSD68 train={len(train_paths)} val={len(val_paths)} "
        f"test={len(test_paths)} missing_rate={args.missing_rate} "
        f"device={device}",
        flush=True,
    )

    factories = {
        "direct_unet": lambda: DirectUNet(args.hidden, args.depth),
        "weight_tied": lambda: WeightTiedUnrolled(
            args.hidden, args.depth, args.steps
        ),
        "non_shared": lambda: NonSharedUnrolled(
            args.hidden, args.depth, args.steps
        ),
        "aenet": lambda: RecurrentAENet(args.hidden, args.depth, args.steps),
        "aenet_v2": lambda: RecurrentAENetV2(
            args.hidden, args.depth, args.steps
        ),
    }
    results = {"dataset": "BSD400/BSD68", "config": vars(args), "models": {}}
    for name in args.models:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(args.seed)
        model = factories[name]()
        start = time.time()
        history, best_val = train_one(
            model,
            train_loader,
            val_loader,
            device,
            args,
        )
        test = evaluate(model, test_loader, device)
        checkpoint = Path(args.output).with_name(
            f"generic_inpainting_{name}_s{args.seed}.pt"
        )
        torch.save(
            {"model": model.state_dict(), "config": vars(args)},
            checkpoint,
        )
        results["models"][name] = {
            "params": count_parameters(model),
            "checkpoint": str(checkpoint),
            "train_sec": round(time.time() - start, 2),
            "best_val": best_val,
            "test": test,
            "history": history,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{name}: test={test}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved to {output}", flush=True)


if __name__ == "__main__":
    main()
