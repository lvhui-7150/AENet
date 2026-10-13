"""Analyze completed fastMRI knee medium-scale model-zoo runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    family: str
    color: str
    linestyle: str = "-"
    official_adapter: bool = False


MODEL_SPECS = [
    ModelSpec(
        "unet_varnet",
        "VarNet non-shared U-Net",
        "VarNet",
        "#ff7f0e",
    ),
    ModelSpec(
        "recurrent_aenet",
        "AENet recurrent U-Net",
        "AENet",
        "#1f77b4",
        ":",
    ),
    ModelSpec(
        "fista",
        "FISTA-style",
        "Classic unrolled",
        "#bcbd22",
        "--",
    ),
    ModelSpec(
        "varnet_shared",
        "VarNet shared CNN",
        "VarNet",
        "#ff7f0e",
        "-.",
    ),
    ModelSpec(
        "dual_domain_aenet",
        "AENet dual-domain",
        "AENet",
        "#1f77b4",
    ),
    ModelSpec(
        "lpd",
        "LPD",
        "Classic unrolled",
        "#bcbd22",
        ":",
    ),
    ModelSpec(
        "adaptive_cnn_aenet",
        "AENet adaptive CNN",
        "AENet",
        "#1f77b4",
        "--",
    ),
    ModelSpec(
        "unet_varnet_shared",
        "VarNet shared U-Net",
        "VarNet",
        "#ff7f0e",
        "--",
    ),
    ModelSpec(
        "recurrent_cnn_aenet",
        "AENet recurrent CNN",
        "AENet",
        "#1f77b4",
        "-.",
    ),
    ModelSpec(
        "feature_image_shared",
        "Feature-image VarNet",
        "Feature-domain",
        "#8c564b",
    ),
    ModelSpec(
        "feature_image_smaprefine_shared",
        "Feature-image + CSME",
        "Feature-domain",
        "#8c564b",
        "--",
    ),
    ModelSpec(
        "fi_attention_shared",
        "FI attention VarNet",
        "Feature-domain",
        "#8c564b",
        ":",
    ),
    ModelSpec(
        "admm",
        "ADMM-style",
        "Classic unrolled",
        "#bcbd22",
    ),
    ModelSpec(
        "smaprefine_shared",
        "VarNet shared + CSME",
        "VarNet + refinement",
        "#e377c2",
    ),
    ModelSpec(
        "modl",
        "MoDL-style",
        "Classic unrolled",
        "#8c8c8c",
    ),
    ModelSpec(
        "direct_unet",
        "Direct U-Net",
        "Direct U-Net",
        "#9467bd",
    ),
    ModelSpec(
        "attention_unet",
        "Attention U-Net",
        "Direct U-Net",
        "#9467bd",
        "--",
    ),
    ModelSpec(
        "ista_soft",
        "ISTA-style",
        "Classic unrolled",
        "#bcbd22",
    ),
    ModelSpec(
        "official_varnet",
        "Official E2E VarNet",
        "Official adapter",
        "#d62728",
        official_adapter=True,
    ),
    ModelSpec(
        "official_humusnet",
        "Official HUMUS-Net",
        "Official adapter",
        "#8c1d18",
        "--",
        official_adapter=True,
    ),
]


def load_runs(result_dir: Path):
    runs = []
    zero_filled = None
    for spec in MODEL_SPECS:
        path = result_dir / f"fastmri_knee_results_hpc_medium_{spec.slug}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_zero = payload.get("zero_filled")
        if current_zero is None:
            raise ValueError(f"Missing zero_filled block: {path}")
        if zero_filled is None:
            zero_filled = current_zero
        model_name = payload["config"]["only"]
        model = payload["models"][model_name]
        test = model["test"]
        runs.append(
            {
                "slug": spec.slug,
                "label": spec.label,
                "family": spec.family,
                "color": spec.color,
                "linestyle": spec.linestyle,
                "official_adapter": spec.official_adapter,
                "valid_run": not spec.official_adapter,
                "beats_zero_filled": test["psnr"] > zero_filled["psnr"],
                "source": str(path),
                "config": payload["config"],
                "params": model["params"],
                "train_sec": model["train_sec"],
                "best_val": model["best_val"],
                "test": test,
                "history": model["history"],
            }
        )
    return runs, zero_filled


def ordered_runs(runs):
    return sorted(runs, key=lambda run: run["test"]["psnr"], reverse=True)


def planned_and_missing_slugs(result_dir: Path):
    generated_dir = (
        result_dir.parent
        / "AENet_fastMRI_Knee_ModelZoo"
        / "hpc"
        / "generated_fastmri_knee_medium"
    )
    if not generated_dir.exists():
        return [], []
    planned = sorted(
        path.stem.replace("run_fastmri_knee_medium_", "")
        for path in generated_dir.glob("run_fastmri_knee_medium_*.sh")
    )
    present = {spec.slug for spec in MODEL_SPECS}
    missing = [slug for slug in planned if slug not in present]
    return planned, missing


def write_summary_csv(runs, zero_filled, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "model",
                "family",
                "valid_run",
                "beats_zero_filled",
                "params",
                "best_epoch",
                "val_psnr",
                "test_psnr",
                "test_psnr_std",
                "test_ssim",
                "test_ssim_std",
                "test_nmse",
                "test_nmse_std",
                "train_hours",
                "mask",
                "seed",
                "epochs",
            ]
        )
        writer.writerow(
            [
                "-",
                "Zero-filled",
                "Baseline",
                1,
                0,
                0,
                0,
                "",
                f"{zero_filled['psnr']:.6f}",
                f"{zero_filled['psnr_std']:.6f}",
                f"{zero_filled['ssim']:.6f}",
                f"{zero_filled['ssim_std']:.6f}",
                f"{zero_filled['nmse']:.6f}",
                f"{zero_filled['nmse_std']:.6f}",
                0,
                "r4_s42",
                42,
                0,
            ]
        )
        for rank, run in enumerate(ordered_runs(runs), start=1):
            test = run["test"]
            writer.writerow(
                [
                    rank,
                    run["label"],
                    run["family"],
                    int(run["valid_run"]),
                    int(run["beats_zero_filled"]),
                    run["params"],
                    run["best_val"]["epoch"],
                    f"{run['best_val']['psnr']:.6f}",
                    f"{test['psnr']:.6f}",
                    f"{test['psnr_std']:.6f}",
                    f"{test['ssim']:.6f}",
                    f"{test['ssim_std']:.6f}",
                    f"{test['nmse']:.6f}",
                    f"{test['nmse_std']:.6f}",
                    f"{run['train_sec'] / 3600:.6f}",
                    run["config"]["mask_key"],
                    run["config"]["seed"],
                    run["config"]["epochs"],
                ]
            )


def markdown_table(runs, zero_filled):
    lines = [
        "| 排名 | 模型 | 有效运行 | 高于零填充 | 参数量 | 最佳轮次 | 测试 PSNR | SSIM | NMSE | 训练小时 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| - | Zero-filled | 基线 | - | - | - | "
            f"{zero_filled['psnr']:.3f} | {zero_filled['ssim']:.4f} | "
            f"{zero_filled['nmse']:.4f} | - |"
        ),
    ]
    for rank, run in enumerate(ordered_runs(runs), start=1):
        test = run["test"]
        lines.append(
            f"| {rank} | {run['label']} | "
            f"{'是' if run['valid_run'] else '否'} | "
            f"{'是' if run['beats_zero_filled'] else '否'} | "
            f"{run['params']:,} | {run['best_val']['epoch']} | "
            f"{test['psnr']:.3f} | {test['ssim']:.4f} | "
            f"{test['nmse']:.4f} | {run['train_sec'] / 3600:.2f} |"
        )
    return "\n".join(lines)


def plot_ranked_metrics(runs, zero_filled, out_path: Path):
    ranking = ordered_runs(runs)
    y = np.arange(len(ranking))
    metrics = [
        ("psnr", "Test PSNR (dB)", zero_filled["psnr"], "{:.3f}"),
        ("ssim", "Test SSIM", zero_filled["ssim"], "{:.4f}"),
        ("nmse", "Test NMSE", zero_filled["nmse"], "{:.4f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), constrained_layout=True)
    for axis, (key, title, baseline, value_format) in zip(axes, metrics):
        values = [run["test"][key] for run in ranking]
        bars = axis.barh(
            y,
            values,
            color=[run["color"] for run in ranking],
            alpha=0.88,
        )
        for bar, run in zip(bars, ranking):
            if run["official_adapter"]:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
            if not run["beats_zero_filled"]:
                bar.set_edgecolor("#7f1d1d")
                bar.set_linewidth(1.2)
        axis.axvline(
            baseline,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
        )
        axis.text(
            baseline,
            -1.05,
            "zero-filled " + value_format.format(baseline),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        axis.set_yticks(y)
        axis.set_yticklabels([run["label"] for run in ranking], fontsize=8)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.grid(axis="x", alpha=0.22)
        for yi, value in zip(y, values):
            axis.text(
                value,
                yi,
                " " + value_format.format(value),
                va="center",
                fontsize=7,
            )
    handles = [
        Patch(color="#1f77b4", label="AENet"),
        Patch(color="#ff7f0e", label="VarNet"),
        Patch(color="#8c564b", label="Feature-domain"),
        Patch(color="#bcbd22", label="Classic unrolled"),
        Patch(color="#e377c2", label="VarNet + refinement"),
        Patch(color="#9467bd", label="Direct U-Net"),
        Patch(color="#d62728", label="Official adapter"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Invalid run"),
        Line2D([0], [0], color="#222222", linestyle="--", label="Zero-filled"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8)
    fig.suptitle("fastMRI knee medium model-zoo test metrics", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_over_zero(runs, zero_filled, out_path: Path):
    ranking = ordered_runs(runs)
    gains = [run["test"]["psnr"] - zero_filled["psnr"] for run in ranking]
    y = np.arange(len(ranking))
    fig, axis = plt.subplots(figsize=(10, 7))
    bars = axis.barh(
        y,
        gains,
        color=[run["color"] for run in ranking],
        alpha=0.88,
    )
    for bar, run in zip(bars, ranking):
        if run["official_adapter"]:
            bar.set_hatch("///")
            bar.set_edgecolor("black")
    axis.axvline(0.0, color="#222222", linewidth=1.2)
    axis.set_yticks(y)
    axis.set_yticklabels([run["label"] for run in ranking], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(f"Test PSNR gain over zero-filled ({zero_filled['psnr']:.3f} dB)")
    axis.set_title("Gain over zero-filled reconstruction")
    axis.grid(axis="x", alpha=0.22)
    for yi, gain in zip(y, gains):
        offset = 0.04 if gain >= 0 else -0.04
        axis.text(
            gain + offset,
            yi,
            f"{gain:+.3f}",
            va="center",
            ha="left" if gain >= 0 else "right",
            fontsize=7,
        )
    axis.set_xlim(min(gains) - 0.65, max(gains) + 0.65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(runs, zero_filled, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for run in runs:
        marker = "s" if run["official_adapter"] else "o"
        axes[0].errorbar(
            run["params"] / 1e6,
            run["test"]["psnr"],
            yerr=run["test"]["psnr_std"],
            fmt=marker,
            color=run["color"],
            markeredgecolor="black",
            markersize=7,
            capsize=2,
        )
        axes[1].scatter(
            run["train_sec"] / 3600,
            run["test"]["psnr"],
            color=run["color"],
            edgecolor="black",
            marker=marker,
            s=55,
        )
    for axis in axes:
        axis.axhline(
            zero_filled["psnr"],
            color="#222222",
            linestyle="--",
            linewidth=1.2,
        )
        axis.grid(alpha=0.22)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (millions, log scale)")
    axes[0].set_ylabel("Test PSNR (dB)")
    axes[0].set_title("Parameter efficiency")
    axes[1].set_xlabel("Wall-clock training time (hours)")
    axes[1].set_ylabel("Test PSNR (dB)")
    axes[1].set_title("Quality versus training cost")

    lookup = {run["slug"]: run for run in runs}
    parameter_notes = {
        "unet_varnet": (6, 7, "left"),
        "recurrent_aenet": (-6, 8, "right"),
        "dual_domain_aenet": (-6, 8, "right"),
        "recurrent_cnn_aenet": (6, -14, "left"),
        "official_varnet": (6, -14, "left"),
        "official_humusnet": (6, 8, "left"),
    }
    for slug, (dx, dy, alignment) in parameter_notes.items():
        run = lookup[slug]
        axes[0].annotate(
            run["label"],
            (run["params"] / 1e6, run["test"]["psnr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=alignment,
            fontsize=7,
        )
    axes[0].text(
        0.03,
        0.78,
        "AENet: 0.17 to 2.64M parameters\n34.75 to 35.01 dB",
        transform=axes[0].transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    time_notes = {
        "unet_varnet": (6, 7, "left"),
        "recurrent_aenet": (-6, 8, "right"),
        "dual_domain_aenet": (-6, 8, "right"),
        "official_varnet": (6, -14, "left"),
        "official_humusnet": (6, 8, "left"),
    }
    for slug, (dx, dy, alignment) in time_notes.items():
        run = lookup[slug]
        axes[1].annotate(
            run["label"],
            (run["train_sec"] / 3600, run["test"]["psnr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=alignment,
            fontsize=7,
        )
    fig.suptitle("Efficiency of the fastMRI knee development model zoo", fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_trajectories(runs, out_path: Path):
    selected = [
        "unet_varnet",
        "recurrent_aenet",
        "fista",
        "varnet_shared",
        "dual_domain_aenet",
        "unet_varnet_shared",
        "recurrent_cnn_aenet",
        "modl",
        "direct_unet",
        "ista_soft",
        "official_varnet",
        "official_humusnet",
    ]
    lookup = {run["slug"]: run for run in runs}
    fig, axis = plt.subplots(figsize=(11, 6.5))
    for slug in selected:
        run = lookup[slug]
        axis.plot(
            [row["epoch"] for row in run["history"]],
            [row["psnr"] for row in run["history"]],
            label=run["label"],
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=1.8,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation PSNR (dB)")
    axis.set_title("fastMRI knee validation trajectories")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def contrast_descriptive(runs, first_slug, second_slug, zero_filled=None):
    lookup = {run["slug"]: run for run in runs}
    first = lookup[first_slug]
    first_value = first["test"]["psnr"]
    first_std = first["test"]["psnr_std"]
    if second_slug == "zero_filled":
        second_value = zero_filled["psnr"]
        second_std = zero_filled["psnr_std"]
    else:
        second = lookup[second_slug]
        second_value = second["test"]["psnr"]
        second_std = second["test"]["psnr_std"]
    pooled = math.sqrt((first_std**2 + second_std**2) / 2.0)
    return {
        "difference": first_value - second_value,
        "standardized": (first_value - second_value) / pooled,
    }


def effect_size_table(runs, zero_filled):
    contrasts = [
        ("AENet recurrent U-Net", "recurrent_aenet", "VarNet non-shared U-Net", "unet_varnet"),
        ("AENet recurrent U-Net", "recurrent_aenet", "AENet recurrent CNN", "recurrent_cnn_aenet"),
        ("VarNet shared CNN", "varnet_shared", "VarNet shared U-Net", "unet_varnet_shared"),
        ("AENet adaptive CNN", "adaptive_cnn_aenet", "AENet dual-domain", "dual_domain_aenet"),
        ("Feature-image + CSME", "feature_image_smaprefine_shared", "Feature-image VarNet", "feature_image_shared"),
        ("FI attention VarNet", "fi_attention_shared", "Feature-image VarNet", "feature_image_shared"),
        ("Direct U-Net", "direct_unet", "VarNet non-shared U-Net", "unet_varnet"),
        ("ISTA-style", "ista_soft", "Zero-filled", "zero_filled"),
        ("Official E2E VarNet", "official_varnet", "Zero-filled", "zero_filled"),
        ("Official HUMUS-Net", "official_humusnet", "Zero-filled", "zero_filled"),
    ]
    lines = [
        "| 对比 | 差值 PSNR (dB) | 差异/合并标准差 | 解释 |",
        "|---|---:|---:|---|",
    ]
    for first_label, first_slug, second_label, second_slug in contrasts:
        result = contrast_descriptive(
            runs, first_slug, second_slug, zero_filled
        )
        if abs(result["difference"]) < 0.05:
            interpretation = "数值上几乎相同"
        elif abs(result["standardized"]) < 0.2:
            interpretation = "小描述性效应"
        elif abs(result["standardized"]) < 0.8:
            interpretation = "中等描述性效应"
        else:
            interpretation = "大描述性效应"
        lines.append(
            f"| {first_label} vs {second_label} | "
            f"{result['difference']:+.4f} | "
            f"{result['standardized']:+.3f} | "
            f"{interpretation} |"
        )
    return "\n".join(lines)


def format_analysis_report(runs, zero_filled, missing_slugs):
    ranking = ordered_runs(runs)
    lookup = {run["slug"]: run for run in runs}
    best = ranking[0]
    best_aenet = next(run for run in ranking if run["family"] == "AENet")
    aenet_runs = [run for run in ranking if run["family"] == "AENet"]
    best_efficient_aenet = next(
        run for run in ranking if run["slug"] == "recurrent_cnn_aenet"
    )
    top_five = ranking[:5]
    missing_text = ", ".join(missing_slugs) if missing_slugs else "未检测到"
    top_five_spread = top_five[0]["test"]["psnr"] - top_five[-1]["test"]["psnr"]
    aenet_spread = aenet_runs[0]["test"]["psnr"] - aenet_runs[-1]["test"]["psnr"]
    param_ratio = best["params"] / best_aenet["params"]
    top_gap = best_aenet["test"]["psnr"] - best["test"]["psnr"]
    feature_gain = (
        lookup["feature_image_smaprefine_shared"]["test"]["psnr"]
        - lookup["feature_image_shared"]["test"]["psnr"]
    )
    attention_gap = (
        lookup["fi_attention_shared"]["test"]["psnr"]
        - lookup["feature_image_shared"]["test"]["psnr"]
    )
    official_gap = (
        lookup["official_varnet"]["test"]["psnr"] - zero_filled["psnr"]
    )
    humus_gap = (
        lookup["official_humusnet"]["test"]["psnr"] - zero_filled["psnr"]
    )
    ista_gap = lookup["ista_soft"]["test"]["psnr"] - zero_filled["psnr"]

    return f"""# fastMRI Knee Results Analysis

## Analysis Question

这组结果用于回答：在当前 fastMRI knee 单线圈 128 x 128 开发协议上，
AENet 各变体、VarNet 各变体和常规基线分别处于什么水平，哪些差异值得继续验证。

## Protocol And Validity

- 当前完成 {len(runs)} / 22 个计划模型；未完成：{missing_text}。
- 所有 JSON 使用同一数据文件、`r4_s42` 掩码、seed 42 和 100 epochs。
- 数据为 fastMRI knee single-coil，128 x 128；现有 prepared tensor 包含
  199 slices，约按 159/20/20 划分，测试集约 20 slices。
- k-space 由 image-domain ground truth 经中心化正交 FFT 重建，不是直接使用
  raw H5 k-space 的最终协议。
- Zero-filled: PSNR {zero_filled['psnr']:.3f} dB，SSIM {zero_filled['ssim']:.4f}，
  NMSE {zero_filled['nmse']:.4f}。
- `Official E2E VarNet` 和 `Official HUMUS-Net` 均低于零填充，
  标记为无效适配运行。
- `ISTA-style` 虽然运行有效，但测试结果
  {lookup['ista_soft']['test']['psnr']:.3f} dB，相对零填充
  {ista_gap:+.4f} dB，实际退化为零填充解。

## Exact Ranking

{markdown_table(runs, zero_filled)}

## Key Findings

1. 当前第一是 **{best['label']}**：
   PSNR {best['test']['psnr']:.3f}，SSIM {best['test']['ssim']:.4f}，
   NMSE {best['test']['nmse']:.4f}，相对零填充
   {best['test']['psnr'] - zero_filled['psnr']:+.3f} dB。
2. 前五名集中在 {top_five[-1]['test']['psnr']:.3f} 到
   {top_five[0]['test']['psnr']:.3f} dB，总跨度仅
   {top_five_spread:.3f} dB。这个差距远小于 slice-level PSNR 标准差，
   因此不能把第一名的领先解释为稳定胜出。
3. 最佳 AENet 是 **{best_aenet['label']}**，PSNR
   {best_aenet['test']['psnr']:.3f}。它比第一名的非共享 U-Net 低
   {abs(top_gap):.3f} dB，但参数量少约 {param_ratio:.1f} 倍。
4. AENet 四个变体的范围只有 {aenet_spread:.3f} dB。
   recurrent U-Net 质量最高；如果优先考虑计算效率，
   recurrent CNN 以约 0.171M 参数达到
   {best_efficient_aenet['test']['psnr']:.3f} dB，是 AENet 内部更节省参数的方案。
5. `Feature-image + CSME` 相对 `Feature-image VarNet` 改变
   {feature_gain:+.4f} dB；`FI attention VarNet` 改变
   {attention_gap:+.4f} dB。两项模块在当前协议下都没有显示出明确增益。
6. `Direct U-Net` 只有 32.229 dB，明显低于展开式模型，支持“数据一致性展开
   比单纯图像域 U-Net 更有效”的解释。
7. `Official E2E VarNet` 只有 {lookup['official_varnet']['test']['psnr']:.3f} dB，
   比零填充低 {abs(official_gap):.3f} dB，应继续按适配问题排查，不能作为论文中的
   官方方法性能。
8. `Official HUMUS-Net` 只有 {lookup['official_humusnet']['test']['psnr']:.3f} dB，
   比零填充低 {abs(humus_gap):.3f} dB；两个官方模型都在早期训练后平台化，
   当前结果不能视为官方方法性能。

## Cross-Dataset Interpretation

- Knee 128 不能与 M4Raw medium 直接做绝对数值比较。Knee 是单线圈、小型
  20-slice 测试集，而且 k-space 由 GT 图像重建；M4Raw 是多线圈、不同 split 和
  不同 forward-model 来源。
- 可以看到的共同趋势是：展开式方法明显优于 direct U-Net；AENet recurrent CNN
  和 recurrent U-Net 在两条数据线上都处于 AENet 第一梯队；复杂的 feature-domain
  或 attention 变体没有稳定优势。
- Knee 上非共享 U-Net VarNet 排名第一，而 M4Raw 上非共享 CNN VarNet 排名第一。
  这说明当前排名对 backbone 和数据集仍然敏感，不能据此推出统一架构排序。

## Supported Claim Candidates

- Claim: 在当前 fastMRI knee 128 单线圈开发集上，展开式方法明显优于 Direct U-Net。
  - Source evidence: Direct U-Net 32.229 dB，对比顶级展开模型约 34.6 到 35.0 dB。
  - Allowed wording: “在 current development protocol 上明显更好”。
  - Forbidden stronger wording: “在 fastMRI knee 上达到 SOTA”。
  - Uncertainty: 20-slice test set，单 seed，单 mask。
  - Next check: 原生 320、官方 raw-H5 split、多 seed、多 mask。
  - Decision: keep.
- Claim: AENet 的 recurrent U-Net、dual-domain、adaptive CNN 和 recurrent CNN
  当前性能接近。
  - Source evidence: 四者 PSNR 34.754 到 35.010 dB，总跨度 0.256 dB。
  - Allowed wording: “数值上处于同一梯队”。
  - Forbidden stronger wording: “recurrent U-Net 显著优于 recurrent CNN”。
  - Uncertainty: 缺少逐 slice paired metrics。
  - Next check: 保存逐 slice 指标并做 paired test。
  - Decision: keep.
- Claim: 两个官方适配器当前运行无效。
  - Source evidence: E2E VarNet 22.076 dB，HUMUS-Net 21.662 dB，
    均低于 zero-filled 28.919 dB。
  - Allowed wording: “当前适配运行无效”。
  - Forbidden stronger wording: “官方方法性能很差”。
  - Uncertainty: 具体适配故障尚未定位。
  - Next check: 检查单线圈接口、内部 sensitivity model 和官方训练设置。
  - Decision: keep.

## Required Next Checks

1. 补齐基础 `aenet` 和 `varnet` 非共享 CNN，完成协议中的 22 个模型。
2. 将 U-Net backbone、参数规模和训练轮次分开分析。
3. 保存逐 slice 指标或预测，执行 paired Wilcoxon 或 paired t-test。
4. 增加 seed 与 mask 重复，最好以 volume 为 bootstrap 单元。
5. 将当前 128 x 128 开发结果与原生 320 或 raw-H5 结果分开报告。
"""


def format_stats_appendix(runs, zero_filled):
    return f"""# Statistics Appendix

## Descriptive Statistics

所有指标均为测试集 slice-level mean。JSON 记录了跨 slice 标准差，
但没有保存逐 slice 数值，也没有 seed-level repeats。

{markdown_table(runs, zero_filled)}

## Descriptive Effect Sizes

下表使用 `mean difference / pooled slice-level standard deviation`。
该比值只用于描述当前均值差异相对于跨 slice 波动的量级，
不是 paired effect size，也不能替代显著性检验。

{effect_size_table(runs, zero_filled)}

## Inferential Status

- Test units: 约 20 个 test slices，来自同一开发数据 split。
- Seeds per model: 1。
- Masks per model: 1。
- Paired slice-level values: 未保存。
- Confidence intervals and p-values: blocked，当前 JSON 不足。
- Multiple comparisons: 未执行，因为没有可用的 inferential tests。

## Why Naive Pooled Variance Is Not Enough

多个 slice 可能来自同一 volume，不能默认视为完全独立样本。
即使补做 slice-level paired test，也应同时报告 volume-level aggregation
或 cluster bootstrap，避免把同一受试者的相关切片当作独立重复。

## Required Raw Data For Strict Inference

- 每个模型逐 slice 的 PSNR、SSIM、NMSE；
- slice 到 volume 的映射；
- 至少两个独立 seed 和两个 mask；
- 模型输出或足以重算指标的重建结果。
"""


def format_figure_catalog():
    return """# Figure Catalog

## figure-01-ranked-test-metrics.png

- Purpose: 比较全部已完成模型的 PSNR、SSIM 和 NMSE。
- Data source: `fastmri_knee_results_hpc_medium_*.json` 的 test 汇总。
- Error bars: 不显示；跨 slice 标准差记录在 `summary.csv`。
- Caption requirement: 注明 20-slice development test set、single seed、
  single mask，以及官方 E2E VarNet 为 invalid adapter run。
- What to notice: 前五名聚集，Official E2E VarNet 位于零填充下方。
- Decision impact: 不应仅凭 0.2 dB 内差异宣布架构胜出。

## figure-02-gain-over-zero-filled.png

- Purpose: 量化每个模型相对零填充的净 PSNR 增益。
- Data source: model test PSNR 与 JSON 内 zero-filled PSNR。
- Error bars: 不显示。
- Caption requirement: 明确零填充与 learned models 使用同一 mask。
- What to notice: ISTA-style 几乎为零增益，official adapter 为大负增益。
- Decision impact: ISTA 不能作为有效 learned baseline。

## figure-03-efficiency.png

- Purpose: 比较参数量、训练时间和重建质量。
- Data source: `params`、`train_sec`、test PSNR。
- Error bars: 左图显示跨 slice PSNR 标准差。
- Caption requirement: 说明训练时间来自该 199-slice 开发集，不能外推到完整数据。
- What to notice: AENet 用约 0.17 到 0.25M 参数达到约 34.8 dB。
- Decision impact: AENet recurrent CNN 是后续扩展时最有计算效率的候选。

## figure-04-validation-trajectories.png

- Purpose: 检查收敛速度和训练稳定性。
- Data source: 每个 JSON 的 100-epoch history。
- Error bars: 无。
- Caption requirement: 注明曲线为 validation PSNR，best checkpoint 用于测试。
- What to notice: 主要模型在 40 到 60 epochs 后进入平台；
  official E2E VarNet 在约 22 dB 附近饱和。
- Decision impact: official adapter 的问题发生在训练早期；
  direct U-Net 的欠拟合不能靠延长当前训练简单解决。
"""


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "analysis-fastmri-knee-v1"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs, zero_filled = load_runs(script_dir)
    _, missing_slugs = planned_and_missing_slugs(script_dir)
    write_summary_csv(runs, zero_filled, out_dir / "summary.csv")
    (out_dir / "analysis-report.md").write_text(
        format_analysis_report(runs, zero_filled, missing_slugs),
        encoding="utf-8",
    )
    (out_dir / "stats-appendix.md").write_text(
        format_stats_appendix(runs, zero_filled),
        encoding="utf-8",
    )
    (out_dir / "figure-catalog.md").write_text(
        format_figure_catalog(),
        encoding="utf-8",
    )
    plot_ranked_metrics(
        runs,
        zero_filled,
        figures_dir / "figure-01-ranked-test-metrics.png",
    )
    plot_gain_over_zero(
        runs,
        zero_filled,
        figures_dir / "figure-02-gain-over-zero-filled.png",
    )
    plot_efficiency(
        runs,
        zero_filled,
        figures_dir / "figure-03-efficiency.png",
    )
    plot_validation_trajectories(
        runs,
        figures_dir / "figure-04-validation-trajectories.png",
    )

    best = ordered_runs(runs)[0]
    print(f"Wrote analysis to {out_dir}")
    print(
        f"Best run: {best['label']} "
        f"PSNR={best['test']['psnr']:.4f} "
        f"SSIM={best['test']['ssim']:.4f} "
        f"NMSE={best['test']['nmse']:.4f}"
    )
    print(f"Missing planned runs: {', '.join(missing_slugs)}")


if __name__ == "__main__":
    main()
