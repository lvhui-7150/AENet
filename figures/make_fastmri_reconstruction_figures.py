"""Create qualitative fastMRI knee reconstruction figures.

The script reconstructs every test slice with the trained AENet recurrent
U-Net and the leading non-shared U-Net VarNet, then saves:

    fastmri_recon_examples.png
    fastmri_recon_error_zoom.png
    fastmri_recon_slice_metrics.png

It also writes PDF versions for LaTeX inclusion.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_aenet_m4raw import (  # noqa: E402
    ComplexUNet,
    RecurrentAENetMC,
    VarNetMC,
    make_inputs,
)
from train_tpami import metric_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--aenet-checkpoint", required=True)
    parser.add_argument("--varnet-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"], checkpoint["config"], checkpoint["mask"]


def build_models(aenet_path: Path, varnet_path: Path):
    aenet_state, aenet_cfg, mask = load_checkpoint(aenet_path)
    varnet_state, varnet_cfg, varnet_mask = load_checkpoint(varnet_path)
    if not torch.equal(mask, varnet_mask):
        raise RuntimeError("AENet and VarNet checkpoints use different masks")

    aenet = RecurrentAENetMC(
        mask,
        steps=int(aenet_cfg["steps"]),
        hidden=int(aenet_cfg["hidden"]),
        depth=int(aenet_cfg["depth"]),
        anchor_init=float(aenet_cfg["anchor_init"]),
    )
    varnet = VarNetMC(
        mask,
        steps=int(varnet_cfg["steps"]),
        hidden=int(varnet_cfg["hidden"]),
        depth=int(varnet_cfg["depth"]),
        prior_factory=ComplexUNet,
    )
    aenet.load_state_dict(aenet_state)
    varnet.load_state_dict(varnet_state)
    return aenet, varnet, mask.float()


@torch.no_grad()
def reconstruct_all(
    data_path: Path,
    aenet: torch.nn.Module,
    varnet: torch.nn.Module,
    mask: torch.Tensor,
    device: torch.device,
):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    gt_all = data["gt_complex"]
    smap_all = data["smap"]
    kspace_all = data["kspace"]
    test_indices = list(data["split"]["test"])

    aenet.eval().to(device)
    varnet.eval().to(device)
    mask = mask.to(device)

    records = []
    for position, index in enumerate(test_indices, start=1):
        gt_batch = gt_all[index : index + 1]
        smap_batch = smap_all[index : index + 1]
        kspace_batch = kspace_all[index : index + 1]
        gt, y, z0 = make_inputs(gt_batch, smap_batch, kspace_batch, mask, device)
        smap = smap_batch.to(device)

        recon_aenet = aenet(z0, y, smap)
        recon_varnet = varnet(z0, y, smap)
        metrics_zero = metric_batch(z0, gt)
        metrics_aenet = metric_batch(recon_aenet, gt)
        metrics_varnet = metric_batch(recon_varnet, gt)

        records.append(
            {
                "index": int(index),
                "gt": gt[0].abs().cpu().numpy(),
                "zero": z0[0].abs().cpu().numpy(),
                "aenet": recon_aenet[0].abs().cpu().numpy(),
                "varnet": recon_varnet[0].abs().cpu().numpy(),
                "zero_metrics": {
                    key: float(value[0]) for key, value in metrics_zero.items()
                },
                "aenet_metrics": {
                    key: float(value[0]) for key, value in metrics_aenet.items()
                },
                "varnet_metrics": {
                    key: float(value[0]) for key, value in metrics_varnet.items()
                },
            }
        )
        print(f"[reconstruct] {position}/{len(test_indices)} index={index}")
    return records


def select_representative_indices(records):
    aenet_psnr = np.asarray([row["aenet_metrics"]["psnr"] for row in records])
    varnet_psnr = np.asarray([row["varnet_metrics"]["psnr"] for row in records])
    median_index = int(np.argsort(aenet_psnr)[len(records) // 2])
    varnet_win = int(np.argmax(varnet_psnr - aenet_psnr))
    aenet_win = int(np.argmax(aenet_psnr - varnet_psnr))
    return [median_index, varnet_win, aenet_win]


def style_axis(axis, title: str):
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def save_figure(fig, out_dir: Path, stem: str):
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def plot_examples(records, out_dir: Path):
    selected = select_representative_indices(records)
    columns = (
        "GT",
        "Zero-filled",
        "AENet",
        "VarNet",
        "AENet error x5",
        "VarNet error x5",
    )
    fig, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(14, 6.8),
        constrained_layout=True,
    )
    for row_index, record_index in enumerate(selected):
        row = records[record_index]
        gt = row["gt"]
        vmax = float(np.percentile(gt, 99.7))
        vmax = max(vmax, 1e-6)
        errors = {
            "aenet": np.abs(row["aenet"] - gt),
            "varnet": np.abs(row["varnet"] - gt),
        }
        err_vmax = max(
            float(np.percentile(errors["aenet"], 99.5)),
            float(np.percentile(errors["varnet"], 99.5)),
            1e-6,
        )
        images = [
            (gt, "gray", vmax),
            (row["zero"], "gray", vmax),
            (row["aenet"], "gray", vmax),
            (row["varnet"], "gray", vmax),
            (5.0 * errors["aenet"], "magma", 5.0 * err_vmax),
            (5.0 * errors["varnet"], "magma", 5.0 * err_vmax),
        ]
        for column_index, (image, cmap, scale) in enumerate(images):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap, vmin=0, vmax=scale)
            if row_index == 0:
                style_axis(axis, columns[column_index])
            else:
                style_axis(axis, "")
        axes[row_index, 0].set_ylabel(
            f"test slice {row['index']}",
            fontsize=8,
        )
        axes[row_index, 2].set_xlabel(
            f"PSNR {row['aenet_metrics']['psnr']:.2f} | "
            f"SSIM {row['aenet_metrics']['ssim']:.3f}",
            fontsize=7,
        )
        axes[row_index, 3].set_xlabel(
            f"PSNR {row['varnet_metrics']['psnr']:.2f} | "
            f"SSIM {row['varnet_metrics']['ssim']:.3f}",
            fontsize=7,
        )
    fig.suptitle(
        "fastMRI knee reconstructions: representative test slices",
        fontsize=12,
    )
    save_figure(fig, out_dir, "fastmri_recon_examples")


def crop_around_error(gt, error, size=56):
    y, x = np.unravel_index(int(np.argmax(error)), error.shape)
    half = size // 2
    top = min(max(y - half, 0), gt.shape[0] - size)
    left = min(max(x - half, 0), gt.shape[1] - size)
    return np.s_[top : top + size], np.s_[left : left + size]


def plot_error_zoom(records, out_dir: Path):
    mean_error = np.asarray(
        [
            np.mean(np.abs(row["aenet"] - row["gt"]))
            + np.mean(np.abs(row["varnet"] - row["gt"]))
            for row in records
        ]
    )
    row = records[int(np.argmax(mean_error))]
    crop = crop_around_error(
        row["gt"],
        np.maximum(
            np.abs(row["aenet"] - row["gt"]),
            np.abs(row["varnet"] - row["gt"]),
        ),
    )
    panels = [
        ("GT", row["gt"][crop], "gray"),
        ("Zero-filled", row["zero"][crop], "gray"),
        ("AENet", row["aenet"][crop], "gray"),
        ("VarNet", row["varnet"][crop], "gray"),
        ("AENet error x5", 5.0 * np.abs(row["aenet"] - row["gt"])[crop], "magma"),
        (
            "VarNet error x5",
            5.0 * np.abs(row["varnet"] - row["gt"])[crop],
            "magma",
        ),
    ]
    gt_vmax = max(float(panels[0][1].max()), 1e-6)
    err_vmax = max(float(panels[4][1].max()), float(panels[5][1].max()), 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8), constrained_layout=True)
    for axis, (title, image, cmap) in zip(axes.ravel(), panels):
        vmax = err_vmax if cmap == "magma" else gt_vmax
        axis.imshow(image, cmap=cmap, vmin=0, vmax=vmax)
        style_axis(axis, title)
    fig.suptitle(
        f"fastMRI knee reconstruction error zoom (test slice {row['index']})",
        fontsize=12,
    )
    save_figure(fig, out_dir, "fastmri_recon_error_zoom")


def plot_slice_metrics(records, out_dir: Path):
    a_psnr = np.asarray([row["aenet_metrics"]["psnr"] for row in records])
    v_psnr = np.asarray([row["varnet_metrics"]["psnr"] for row in records])
    a_ssim = np.asarray([row["aenet_metrics"]["ssim"] for row in records])
    v_ssim = np.asarray([row["varnet_metrics"]["ssim"] for row in records])
    x = np.arange(len(records))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot(x, v_psnr, "o-", label="VarNet", color="#ff7f0e", markersize=4)
    axes[0].plot(x, a_psnr, "s-", label="AENet", color="#1f77b4", markersize=4)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Per-slice PSNR")
    axes[1].plot(x, v_ssim, "o-", label="VarNet", color="#ff7f0e", markersize=4)
    axes[1].plot(x, a_ssim, "s-", label="AENet", color="#1f77b4", markersize=4)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Per-slice SSIM")
    for axis in axes:
        axis.set_xlabel("Test slice")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].text(
        0.02,
        0.98,
        f"AENet mean {a_psnr.mean():.3f} dB\n"
        f"VarNet mean {v_psnr.mean():.3f} dB",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].text(
        0.02,
        0.98,
        f"AENet mean {a_ssim.mean():.4f}\n"
        f"VarNet mean {v_ssim.mean():.4f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    fig.suptitle("fastMRI knee paired reconstruction metrics", fontsize=12)
    save_figure(fig, out_dir, "fastmri_recon_slice_metrics")


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aenet, varnet, mask = build_models(
        Path(args.aenet_checkpoint),
        Path(args.varnet_checkpoint),
    )
    records = reconstruct_all(
        Path(args.data),
        aenet,
        varnet,
        mask,
        device,
    )
    plot_examples(records, out_dir)
    plot_error_zoom(records, out_dir)
    plot_slice_metrics(records, out_dir)


if __name__ == "__main__":
    main()
