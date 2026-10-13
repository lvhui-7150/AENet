"""Create a publication-ready visual comparison for BSD68 inpainting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "AENet_generic_patch_20261009_v3" / "experiments" / "train_generic_inpainting.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("generic_training", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_result_metrics() -> dict[str, dict[str, float]]:
    safe = json.loads(
        (ROOT / "results" / "generic_inpainting_bsd_safe_results.json").read_text(
            encoding="utf-8"
        )
    )
    large = json.loads(
        (ROOT / "results" / "generic_inpainting_aenet_v2_large_results.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = {
        "direct_unet": safe["models"]["direct_unet"]["test"],
        "weight_tied": safe["models"]["weight_tied"]["test"],
        "non_shared": safe["models"]["non_shared"]["test"],
        "aenet_v2_large": large["models"]["aenet_v2"]["test"],
    }
    return metrics


def make_model(training, kind: str, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    hidden = int(config["hidden"])
    depth = int(config["depth"])
    steps = int(config["steps"])
    factories = {
        "direct_unet": lambda: training.DirectUNet(hidden, depth),
        "weight_tied": lambda: training.WeightTiedUnrolled(hidden, depth, steps),
        "non_shared": lambda: training.NonSharedUnrolled(hidden, depth, steps),
        "aenet_v2_large": lambda: training.RecurrentAENetV2(hidden, depth, steps),
    }
    model = factories[kind]().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def grayscale(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 1:
        return np.clip(array[0], 0.0, 1.0)
    return np.clip(
        0.2126 * array[0] + 0.7152 * array[1] + 0.0722 * array[2],
        0.0,
        1.0,
    )


def infer_sample(training, models, dataset, index: int, device: torch.device):
    batch = dataset[index]
    image = batch["image"].unsqueeze(0).to(device)
    observed = batch["observed"].unsqueeze(0).to(device)
    mask = batch["mask"].unsqueeze(0).to(device)
    outputs = {}
    with torch.no_grad():
        for name, model in models.items():
            outputs[name] = model(observed, mask)
    return {
        "ground_truth": grayscale(image),
        "masked_input": grayscale(observed),
        **{name: grayscale(output) for name, output in outputs.items()},
    }


def metric_label(metrics: dict[str, float]) -> str:
    return f"{metrics['psnr']:.3f} dB\nSSIM {metrics['ssim']:.4f}"


def create_figure(samples: list[dict], sample_names: list[str], metrics: dict[str, dict[str, float]], output_stem: Path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    columns = [
        ("ground_truth", "Ground truth", None),
        ("masked_input", "Masked input\n(85% missing)", None),
        ("direct_unet", "Direct U-Net", metrics["direct_unet"]),
        ("weight_tied", "Weight-tied\nunrolled", metrics["weight_tied"]),
        ("non_shared", "Non-shared\nunrolled", metrics["non_shared"]),
        ("aenet_v2_large", "AENet v2-large\n(capacity variant)", metrics["aenet_v2_large"]),
    ]
    n_rows = len(samples)
    n_cols = len(columns)
    # Square 180x180 panels: keep the canvas height close to n_rows * panel
    # width so the grid has no vertical slack between sample rows.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.16, 3.7),
        squeeze=False,
        gridspec_kw={"wspace": 0.045, "hspace": 0.16},
    )
    for row_index, (sample, sample_name) in enumerate(zip(samples, sample_names)):
        for col_index, (key, title, metric) in enumerate(columns):
            axis = axes[row_index, col_index]
            axis.imshow(sample[key], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("#777777")
            if row_index == 0:
                label = title if metric is None else f"{title}\n{metric_label(metric)}"
                axis.set_title(label, fontsize=6.3, pad=3.0, linespacing=1.15)
        axes[row_index, 0].set_ylabel(
            sample_name,
            rotation=90,
            fontsize=6.8,
            labelpad=4.0,
            va="center",
        )

    fig.text(
        0.5,
        0.012,
        "All columns use the same BSD68 image crop and fixed random mask; values above model columns are held-out test means.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "inpainting" / "raw" / "denoising-datasets-main",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--sample-names",
        nargs="+",
        default=["test001.png", "test010.png", "test068.png"],
    )
    args = parser.parse_args()

    training = load_training_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = args.data_root / "BSD68" / "original"
    paths = [test_root / name for name in args.sample_names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing BSD68 samples: " + ", ".join(missing))

    dataset = training.BSDInpaintingDataset(
        test_root,
        paths,
        image_size=180,
        missing_rate=0.85,
        mask_seed=42 + 2000003,
        training=False,
        augment=False,
    )
    checkpoint_paths = {
        "direct_unet": args.checkpoint_dir / "generic_inpainting_direct_unet_s42.pt",
        "weight_tied": args.checkpoint_dir / "generic_inpainting_weight_tied_s42.pt",
        "non_shared": args.checkpoint_dir / "generic_inpainting_non_shared_s42.pt",
        "aenet_v2_large": args.checkpoint_dir / "generic_inpainting_aenet_v2_s42.pt",
    }
    models = {
        name: make_model(training, name, checkpoint, device)
        for name, checkpoint in checkpoint_paths.items()
    }
    samples = [infer_sample(training, models, dataset, index, device) for index in range(len(dataset))]
    metrics = load_result_metrics()
    sample_labels = [Path(name).stem.replace("test", "BSD68 #") for name in args.sample_names]
    create_figure(samples, sample_labels, metrics, args.out_dir / "generic_inpainting_comparison")
    print(f"saved {args.out_dir / 'generic_inpainting_comparison.pdf'}")
    print(f"device={device}; samples={', '.join(args.sample_names)}")


if __name__ == "__main__":
    main()
