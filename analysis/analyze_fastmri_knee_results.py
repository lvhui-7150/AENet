"""Rank and plot completed fastMRI knee model-zoo results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "aenet": "AENet CNN",
    "varnet_shared": "VarNet shared CNN",
    "varnet": "VarNet non-shared CNN",
    "recurrent_aenet": "AENet recurrent U-Net",
    "unet_varnet_shared": "VarNet shared U-Net",
    "unet_varnet": "VarNet non-shared U-Net",
    "recurrent_cnn_aenet": "AENet recurrent CNN",
    "adaptive_cnn_aenet": "AENet adaptive CNN",
    "dual_domain_aenet": "AENet dual-domain",
    "feature_image_shared": "Feature-image VarNet",
    "fi_attention_shared": "FI attention VarNet",
    "smaprefine_shared": "VarNet shared + CSME",
    "feature_image_smaprefine_shared": "Feature-image + CSME",
    "official_varnet": "Official E2E VarNet",
    "official_humusnet": "Official HUMUS-Net",
    "direct_unet": "Direct U-Net",
    "modl": "MoDL-style",
    "ista_soft": "ISTA-style",
    "fista": "FISTA-style",
    "admm": "ADMM-style",
    "lpd": "LPD",
    "attention_unet": "Attention U-Net",
}


def infer_slug(path: Path) -> str:
    name = path.stem
    markers = (
        "fastmri_knee_results_hpc_medium_",
        "fastmri_knee_results_",
    )
    for marker in markers:
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def load_runs(result_dir: Path, pattern: str):
    runs = {}
    for path in sorted(result_dir.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_name = payload["config"]["only"]
        if model_name not in payload["models"]:
            continue
        slug = infer_slug(path)
        model = payload["models"][model_name]
        zero_filled = payload.get("zero_filled")
        valid = True
        if slug == "official_varnet" and zero_filled is not None:
            valid = model["test"]["psnr"] >= zero_filled["psnr"]
        runs[LABELS.get(slug, slug)] = {
            "valid": valid,
            "params": model["params"],
            "best_val": model["best_val"],
            "test": model["test"],
            "history": model["history"],
            "source": str(path),
            "zero_filled": zero_filled,
        }
    if not runs:
        raise RuntimeError(f"no result JSON files matched {result_dir / pattern}")
    return runs


def ranked_table(runs):
    order = sorted(runs, key=lambda name: runs[name]["test"]["psnr"], reverse=True)
    lines = [
        "| Rank | Model | Valid | Params | Test PSNR | Test SSIM | Test NMSE |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, label in enumerate(order, start=1):
        run = runs[label]
        test = run["test"]
        lines.append(
            f"| {rank} | {label} | {'yes' if run['valid'] else 'NO'} | "
            f"{run['params']:,} | {test['psnr']:.3f} | "
            f"{test['ssim']:.4f} | {test['nmse']:.4f} |"
        )
    return "\n".join(lines), order


def plot_ranked(runs, out_dir: Path):
    order = sorted(runs, key=lambda name: runs[name]["test"]["psnr"], reverse=True)
    metrics = (
        ("psnr", "Test PSNR (dB)"),
        ("ssim", "Test SSIM"),
        ("nmse", "Test NMSE"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    y = np.arange(len(order))
    for axis, (key, title) in zip(axes, metrics):
        values = [runs[label]["test"][key] for label in order]
        colors = [
            "#2ca02c" if runs[label]["valid"] else "#d62728" for label in order
        ]
        axis.barh(y, values, color=colors, alpha=0.88)
        axis.set_yticks(y)
        axis.set_yticklabels(order, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.grid(axis="x", alpha=0.25)
        for yi, value in enumerate(values):
            axis.text(value, yi, f" {value:.3f}", va="center", fontsize=7)
    fig.suptitle("fastMRI knee model-zoo test results", fontsize=13)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-01-ranked-test-metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_efficiency(runs, out_dir: Path):
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for label, run in runs.items():
        axis.errorbar(
            run["params"] / 1e3,
            run["test"]["psnr"],
            yerr=run["test"]["psnr_std"],
            fmt="o",
            color="#2ca02c" if run["valid"] else "#d62728",
            markeredgecolor="black",
            markersize=7,
            capsize=3,
        )
        axis.annotate(
            label,
            (run["params"] / 1e3, run["test"]["psnr"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (thousands, log scale)")
    axis.set_ylabel("Test PSNR (dB)")
    axis.set_title("fastMRI knee parameter efficiency")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-02-parameter-efficiency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_curves(runs, out_dir: Path):
    selected = (
        "VarNet non-shared CNN",
        "VarNet shared CNN",
        "AENet recurrent CNN",
        "AENet dual-domain",
        "Feature-image VarNet",
        "VarNet shared + CSME",
    )
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for label in selected:
        if label not in runs:
            continue
        history = runs[label]["history"]
        axis.plot(
            [row["epoch"] for row in history],
            [row["psnr"] for row in history],
            label=label,
            linewidth=1.7,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation PSNR (dB)")
    axis.set_title("fastMRI knee validation trajectories")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-03-selected-validation-curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_outputs(runs, out_dir: Path):
    table, order = ranked_table(runs)
    best = order[0]
    zero = next(
        (run["zero_filled"] for run in runs.values() if run["zero_filled"]),
        None,
    )
    zero_line = (
        f"- Zero-filled control: {zero['psnr']:.3f} dB.\n" if zero is not None else ""
    )
    report = f"""# fastMRI Knee Model-Zoo Results

## Ranking

{table}

## Summary

- Best valid model: **{best}**, {runs[best]['test']['psnr']:.3f} dB.
{zero_line}
## Evidence Limits

- Single seed, one R=4 mask, and slice-level aggregation.
- The test split is carved from the official validation volumes.
- A result is not a stable SOTA claim without multiple seeds and paired tests.
"""
    (out_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "valid",
                "params",
                "best_epoch",
                "test_psnr",
                "test_ssim",
                "test_nmse",
            ]
        )
        if zero is not None:
            writer.writerow(
                [
                    "Zero-filled",
                    1,
                    0,
                    0,
                    f"{zero['psnr']:.6f}",
                    f"{zero['ssim']:.6f}",
                    f"{zero['nmse']:.6f}",
                ]
            )
        for label in order:
            run = runs[label]
            writer.writerow(
                [
                    label,
                    int(run["valid"]),
                    run["params"],
                    run["best_val"]["epoch"],
                    f"{run['test']['psnr']:.6f}",
                    f"{run['test']['ssim']:.6f}",
                    f"{run['test']['nmse']:.6f}",
                ]
            )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--pattern",
        default="fastmri_knee_results_hpc_medium_*.json",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    runs = load_runs(result_dir, args.pattern)
    paths = [
        plot_ranked(runs, out_dir),
        plot_parameter_efficiency(runs, out_dir),
        plot_selected_curves(runs, out_dir),
    ]
    csv_path = write_outputs(runs, out_dir)
    print(csv_path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
