"""Create a many-model fastMRI reconstruction comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_aenet_m4raw import (
    ComplexUNet,
    DirectUNetMC,
    FeatureImageVarNetMC,
    FISTAStyleMC,
    LPDNetMC,
    RecurrentAENetMC,
    VarNetMC,
    make_inputs,
)
from train_tpami import metric_batch


MODEL_SPECS = (
    ("AENet v2", "fastmri_knee_medium_aenet_v2_s123_r4_s42_aenet_s123.pt"),
    ("VarNet", "fastmri_knee_medium_unet_varnet_r4_s42_varnet_s42.pt"),
    ("FISTA", "fastmri_knee_medium_fista_r4_s42_baseline_s42.pt"),
    ("LPD", "fastmri_knee_medium_lpd_r4_s42_baseline_s42.pt"),
    ("Feature-image", "fastmri_knee_medium_feature_image_shared_r4_s42_varnet_shared_s42.pt"),
    ("Direct U-Net", "fastmri_knee_medium_direct_unet_r4_s42_baseline_s42.pt"),
)


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"], payload["config"], payload["mask"]


def build_model(path: Path):
    state, config, mask = load_checkpoint(path)
    arch = config["arch"]
    if arch == "recurrent":
        model = RecurrentAENetMC(
            mask,
            steps=int(config["steps"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
            anchor_init=float(config["anchor_init"]),
        )
    elif arch == "unet":
        model = VarNetMC(
            mask,
            steps=int(config["steps"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
            prior_factory=ComplexUNet,
        )
    elif arch == "fista_style":
        model = FISTAStyleMC(
            mask,
            steps=int(config["steps"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
        )
    elif arch == "lpd_net":
        model = LPDNetMC(
            mask,
            steps=int(config["steps"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
        )
    elif arch == "feature_image_varnet":
        model = FeatureImageVarNetMC(
            mask,
            steps=int(config["steps"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
            feat_ch=int(config["feature_channels"]),
        )
    elif arch == "direct_unet":
        model = DirectUNetMC(
            mask,
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
        )
    else:
        raise ValueError(f"unsupported architecture: {arch}")
    model.load_state_dict(state)
    return model, mask.float()


@torch.no_grad()
def reconstruct_all(data_path: Path, checkpoint_dir: Path, device: torch.device):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    test_indices = list(data["split"]["test"])
    outputs = {"gt": [], "zero": [], "test_indices": test_indices, "metrics": {}}
    first = True

    for label, checkpoint_name in MODEL_SPECS:
        model, mask = build_model(checkpoint_dir / checkpoint_name)
        model.eval().to(device)
        mask = mask.to(device)
        model_images = []
        model_metrics = []
        for index in test_indices:
            gt_batch = data["gt_complex"][index : index + 1]
            smap_batch = data["smap"][index : index + 1]
            kspace_batch = data["kspace"][index : index + 1]
            gt, y, z0 = make_inputs(gt_batch, smap_batch, kspace_batch, mask, device)
            recon = model(z0, y, smap_batch.to(device))
            if first:
                outputs["gt"].append(gt[0].abs().cpu().numpy())
                outputs["zero"].append(z0[0].abs().cpu().numpy())
            model_images.append(recon[0].abs().cpu().numpy())
            metrics = metric_batch(recon, gt)
            model_metrics.append(
                {key: float(value[0]) for key, value in metrics.items()}
            )
        outputs["metrics"][label] = model_metrics
        outputs[label] = model_images
        first = False
        print(f"[reconstruct] {label}")
    return outputs


def select_slices(outputs):
    v2 = np.asarray([row["psnr"] for row in outputs["metrics"]["AENet v2"]])
    varnet = np.asarray([row["psnr"] for row in outputs["metrics"]["VarNet"]])
    difference = v2 - varnet
    order = np.argsort(difference)
    selected = [
        int(order[-1]),
        int(order[len(order) // 2]),
        int(order[0]),
    ]
    return selected


def save(fig, out_dir: Path, stem: str):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def make_comparison(outputs, out_dir: Path):
    selected = select_slices(outputs)
    labels = ["GT", "Zero-fill"] + [label for label, _ in MODEL_SPECS]
    fig, axes = plt.subplots(
        len(selected),
        len(labels),
        figsize=(19.5, 6.8),
        constrained_layout=True,
    )
    for row, slice_index in enumerate(selected):
        gt = outputs["gt"][slice_index]
        vmax = max(float(np.percentile(gt, 99.7)), 1e-6)
        images = [gt, outputs["zero"][slice_index]]
        for label, _ in MODEL_SPECS:
            images.append(outputs[label][slice_index])
        for column, (label, image) in enumerate(zip(labels, images)):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(label, fontsize=8)
            if column >= 2:
                metrics = outputs["metrics"][label][slice_index]
                axis.set_xlabel(
                    f"{metrics['psnr']:.2f} dB | {metrics['ssim']:.3f}",
                    fontsize=6,
                )
        axes[row, 0].set_ylabel(
            f"slice {outputs['test_indices'][slice_index]}",
            fontsize=8,
        )
    fig.suptitle(
        "fastMRI knee reconstruction across model families",
        fontsize=13,
    )
    save(fig, out_dir, "fastmri_multimodel_reconstruction")

    with (out_dir / "fastmri_multimodel_slice_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice_index", "model", "psnr", "ssim", "nmse"])
        for slice_index in selected:
            for label, _ in MODEL_SPECS:
                metrics = outputs["metrics"][label][slice_index]
                writer.writerow(
                    [
                        outputs["test_indices"][slice_index],
                        label,
                        metrics["psnr"],
                        metrics["ssim"],
                        metrics["nmse"],
                    ]
                )


def make_error_maps(outputs, out_dir: Path):
    selected = select_slices(outputs)
    labels = ["AENet v2", "VarNet", "FISTA", "Direct U-Net"]
    fig, axes = plt.subplots(
        len(selected),
        len(labels),
        figsize=(12.5, 7.0),
        constrained_layout=True,
    )
    for row, slice_index in enumerate(selected):
        gt = outputs["gt"][slice_index]
        errors = []
        for label in labels:
            errors.append(5.0 * np.abs(outputs[label][slice_index] - gt))
        vmax = max(float(np.percentile(error, 99.7)) for error in errors)
        vmax = max(vmax, 1e-6)
        for column, (label, error) in enumerate(zip(labels, errors)):
            axis = axes[row, column]
            axis.imshow(error, cmap="magma", vmin=0, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"{label} error x5", fontsize=8)
        axes[row, 0].set_ylabel(
            f"slice {outputs['test_indices'][slice_index]}",
            fontsize=8,
        )
    fig.suptitle("fastMRI knee reconstruction error maps", fontsize=13)
    save(fig, out_dir, "fastmri_multimodel_error_maps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = reconstruct_all(
        Path(args.data),
        Path(args.checkpoint_dir),
        device,
    )
    make_comparison(outputs, out_dir)
    make_error_maps(outputs, out_dir)


if __name__ == "__main__":
    main()
