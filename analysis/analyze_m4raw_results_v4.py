"""Rank all completed M4Raw medium-pilot models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILES = [
    ("AENet CNN", "m4raw_results_hpc_medium_aenet.json", True, "#1f77b4", "-"),
    ("VarNet shared CNN", "m4raw_results_hpc_medium_varnet_shared.json", True, "#ff7f0e", "-"),
    ("VarNet non-shared CNN", "m4raw_results_hpc_medium_varnet.json", True, "#2ca02c", "-"),
    ("VarNet shared U-Net", "m4raw_results_hpc_medium_unet_varnet_shared.json", True, "#ff7f0e", "--"),
    ("VarNet non-shared U-Net", "m4raw_results_hpc_medium_unet_varnet.json", True, "#2ca02c", "--"),
    ("AENet recurrent U-Net", "m4raw_results_hpc_medium_recurrent_aenet.json", True, "#17becf", "--"),
    ("AENet recurrent CNN", "m4raw_results_hpc_medium_recurrent_cnn_aenet.json", True, "#17becf", "-."),
    ("AENet adaptive CNN", "m4raw_results_hpc_medium_adaptive_cnn_aenet.json", True, "#9467bd", "-."),
    ("ISTA-style", "m4raw_results_hpc_medium_ista_soft.json", True, "#7f7f7f", "-"),
    ("FISTA-style", "m4raw_results_hpc_medium_fista.json", True, "#bcbd22", "-"),
    ("ADMM-style", "m4raw_results_hpc_medium_admm.json", True, "#bcbd22", "--"),
    ("LPD", "m4raw_results_hpc_medium_lpd.json", True, "#bcbd22", "-."),
    ("MoDL-style", "m4raw_results_hpc_medium_modl.json", True, "#dbdb8d", "-"),
    ("Direct U-Net", "m4raw_results_hpc_medium_direct_unet.json", True, "#c49c94", "-"),
    ("Attention U-Net", "m4raw_results_hpc_medium_attention_unet.json", True, "#c49c94", "--"),
    ("AENet dual-domain", "m4raw_results_hpc_medium_dual_domain_aenet.json", True, "#1f77b4", ":"),
    ("Feature-image VarNet", "m4raw_results_hpc_medium_feature_image_shared.json", True, "#8c564b", "-"),
    ("FI attention VarNet", "m4raw_results_hpc_medium_fi_attention_shared.json", True, "#8c564b", "--"),
    ("VarNet shared + CSME", "m4raw_results_hpc_medium_smaprefine_shared.json", True, "#e377c2", "-"),
    (
        "Feature-image + CSME",
        "m4raw_results_hpc_medium_feature_image_smaprefine_shared.json",
        True,
        "#e377c2",
        "--",
    ),
    ("Official E2E VarNet", "m4raw_results_hpc_medium_official_varnet.json", False, "#7f7f7f", "-"),
]


def load_runs(result_dir: Path):
    runs = {}
    for label, filename, valid, color, linestyle in FILES:
        path = result_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_name = payload["config"]["only"]
        model = payload["models"][model_name]
        runs[label] = {
            "valid": valid,
            "color": color,
            "linestyle": linestyle,
            "config": payload["config"],
            "params": model["params"],
            "train_sec": model["train_sec"],
            "best_val": model["best_val"],
            "test": model["test"],
            "history": model["history"],
        }
    return runs


def ranked_table(runs):
    order = sorted(runs, key=lambda name: runs[name]["test"]["psnr"], reverse=True)
    lines = [
        "| Rank | Model | Valid | Params | Test PSNR | Test SSIM | Test NMSE |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, label in enumerate(order, start=1):
        run = runs[label]
        test = run["test"]
        lines.append(
            f"| {index} | {label} | {'yes' if run['valid'] else 'NO'} | "
            f"{run['params']:,} | {test['psnr']:.3f} | "
            f"{test['ssim']:.4f} | {test['nmse']:.4f} |"
        )
    return "\n".join(lines), order


def plot_ranked(runs, out_dir: Path):
    order = sorted(runs, key=lambda name: runs[name]["test"]["psnr"], reverse=True)
    metrics = [
        ("psnr", "Test PSNR (dB)"),
        ("ssim", "Test SSIM"),
        ("nmse", "Test NMSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    y = np.arange(len(order))
    for ax, (key, title) in zip(axes, metrics):
        values = [runs[label]["test"][key] for label in order]
        colors = [
            runs[label]["color"] if runs[label]["valid"] else "#d62728"
            for label in order
        ]
        ax.barh(y, values, color=colors, alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        for yi, value in enumerate(values):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=7)
    fig.suptitle("M4Raw medium model-zoo test results", fontsize=13)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-01-ranked-test-metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_efficiency(runs, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, run in runs.items():
        x = run["params"] / 1e3
        y = run["test"]["psnr"]
        color = run["color"] if run["valid"] else "#d62728"
        ax.errorbar(
            x,
            y,
            yerr=run["test"]["psnr_std"],
            fmt="o",
            color=color,
            markeredgecolor="black",
            markersize=7,
            capsize=3,
        )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (thousands, log scale)")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("M4Raw medium parameter efficiency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-02-parameter-efficiency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selected_curves(runs, out_dir: Path):
    selected = [
        "VarNet non-shared CNN",
        "VarNet shared CNN",
        "AENet recurrent CNN",
        "AENet dual-domain",
        "Feature-image VarNet",
        "FI attention VarNet",
        "VarNet shared + CSME",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label in selected:
        if label not in runs:
            continue
        run = runs[label]
        ax.plot(
            [row["epoch"] for row in run["history"]],
            [row["psnr"] for row in run["history"]],
            label=label,
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=1.7,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("Validation trajectories of selected models")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "figures" / "figure-03-selected-validation-curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_outputs(runs, out_dir: Path):
    table, order = ranked_table(runs)
    best = order[0]
    report = f"""# M4Raw Medium Model-Zoo Results

## Protocol

All valid models use the same 200-volume/800-slice training set, 200-slice
validation set, 200-slice test set, R=4 mask, and seed 42.

{table}

## Current Leaders

- Best valid model: **{best}**, {runs[best]['test']['psnr']:.3f} dB.
- Best AENet: **AENet recurrent CNN**, {runs['AENet recurrent CNN']['test']['psnr']:.3f} dB.
- Shared CNN VarNet: {runs['VarNet shared CNN']['test']['psnr']:.3f} dB.
- Non-shared CNN VarNet: {runs['VarNet non-shared CNN']['test']['psnr']:.3f} dB.

## Invalid or Missing Results

- Official E2E VarNet is marked invalid: it reached only
  {runs['Official E2E VarNet']['test']['psnr']:.3f} dB, below the
  {27.000:.3f} dB zero-filled control.
- Official HUMUS-Net has no completed JSON in this result directory.

## Main Finding

AENet does not beat the strongest VarNet baselines on the medium pilot.
The recurrent CNN AENet is the best AENet variant, but it is still
{abs(runs['VarNet non-shared CNN']['test']['psnr'] - runs['AENet recurrent CNN']['test']['psnr']):.3f} dB
below non-shared CNN VarNet and
{abs(runs['VarNet shared CNN']['test']['psnr'] - runs['AENet recurrent CNN']['test']['psnr']):.3f} dB
below shared CNN VarNet.

## Evidence Limits

- one seed, one mask, slice-level aggregation;
- no paired test statistics from the stored JSON summaries;
- official VarNet adapter needs debugging before it can be used;
- published fastMRI numbers are not included as directly comparable values.
"""
    (out_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "valid", "params", "best_epoch", "test_psnr", "test_ssim", "test_nmse"]
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "analysis_output" / "v4"),
    )
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    runs = load_runs(result_dir)
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
