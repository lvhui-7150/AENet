"""Analyze all completed M4Raw medium-scale model-zoo runs."""

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


ZERO_FILLED = {
    "label": "Zero-filled",
    "psnr": 27.00,
    "ssim": 0.651,
    "nmse": 0.0531,
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    filename: str
    family: str
    color: str
    linestyle: str = "-"
    official_adapter: bool = False


MODEL_SPECS = [
    ModelSpec(
        "VarNet non-shared CNN",
        "m4raw_results_hpc_medium_varnet.json",
        "VarNet",
        "#ff7f0e",
    ),
    ModelSpec(
        "VarNet shared CNN",
        "m4raw_results_hpc_medium_varnet_shared.json",
        "VarNet",
        "#ff7f0e",
        "--",
    ),
    ModelSpec(
        "AENet CNN",
        "m4raw_results_hpc_medium_aenet.json",
        "AENet",
        "#1f77b4",
    ),
    ModelSpec(
        "AENet recurrent CNN",
        "m4raw_results_hpc_medium_recurrent_cnn_aenet.json",
        "AENet",
        "#1f77b4",
        "-.",
    ),
    ModelSpec(
        "AENet recurrent U-Net",
        "m4raw_results_hpc_medium_recurrent_aenet.json",
        "AENet",
        "#1f77b4",
        ":",
    ),
    ModelSpec(
        "AENet adaptive CNN",
        "m4raw_results_hpc_medium_adaptive_cnn_aenet.json",
        "AENet",
        "#1f77b4",
        "--",
    ),
    ModelSpec(
        "AENet dual-domain",
        "m4raw_results_hpc_medium_dual_domain_aenet.json",
        "AENet",
        "#1f77b4",
        (0, (3, 1, 1, 1)),
    ),
    ModelSpec(
        "VarNet non-shared U-Net",
        "m4raw_results_hpc_medium_unet_varnet.json",
        "VarNet",
        "#ff7f0e",
        ":",
    ),
    ModelSpec(
        "VarNet shared U-Net",
        "m4raw_results_hpc_medium_unet_varnet_shared.json",
        "VarNet",
        "#ff7f0e",
        (0, (3, 1, 1, 1)),
    ),
    ModelSpec(
        "VarNet shared + CSME",
        "m4raw_results_hpc_medium_smaprefine_shared.json",
        "VarNet + refinement",
        "#e377c2",
    ),
    ModelSpec(
        "Feature-image VarNet",
        "m4raw_results_hpc_medium_feature_image_shared.json",
        "Feature-domain",
        "#8c564b",
    ),
    ModelSpec(
        "FI attention VarNet",
        "m4raw_results_hpc_medium_fi_attention_shared.json",
        "Feature-domain",
        "#8c564b",
        "--",
    ),
    ModelSpec(
        "Feature-image + CSME",
        "m4raw_results_hpc_medium_feature_image_smaprefine_shared.json",
        "Feature-domain",
        "#8c564b",
        ":",
    ),
    ModelSpec(
        "ISTA-style",
        "m4raw_results_hpc_medium_ista_soft.json",
        "Classic unrolled",
        "#bcbd22",
    ),
    ModelSpec(
        "FISTA-style",
        "m4raw_results_hpc_medium_fista.json",
        "Classic unrolled",
        "#bcbd22",
        "--",
    ),
    ModelSpec(
        "ADMM-style",
        "m4raw_results_hpc_medium_admm.json",
        "Classic unrolled",
        "#bcbd22",
        "-.",
    ),
    ModelSpec(
        "LPD",
        "m4raw_results_hpc_medium_lpd.json",
        "Classic unrolled",
        "#bcbd22",
        ":",
    ),
    ModelSpec(
        "MoDL-style",
        "m4raw_results_hpc_medium_modl.json",
        "Classic unrolled",
        "#8c8c8c",
    ),
    ModelSpec(
        "Direct U-Net",
        "m4raw_results_hpc_medium_direct_unet.json",
        "Direct U-Net",
        "#9467bd",
    ),
    ModelSpec(
        "Attention U-Net",
        "m4raw_results_hpc_medium_attention_unet.json",
        "Direct U-Net",
        "#9467bd",
        "--",
    ),
    ModelSpec(
        "Official E2E VarNet",
        "m4raw_results_hpc_medium_official_varnet.json",
        "Official adapter",
        "#d62728",
        official_adapter=True,
    ),
    ModelSpec(
        "Official HUMUS-Net",
        "m4raw_results_hpc_medium_official_humusnet.json",
        "Official adapter",
        "#8c1d18",
        "--",
        official_adapter=True,
    ),
]


def load_runs(result_dir: Path):
    runs = []
    for spec in MODEL_SPECS:
        path = result_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_name = payload["config"]["only"]
        model = payload["models"][model_name]
        test = model["test"]
        best_val = model["best_val"]
        runs.append(
            {
                "label": spec.label,
                "filename": spec.filename,
                "family": spec.family,
                "color": spec.color,
                "linestyle": spec.linestyle,
                "official_adapter": spec.official_adapter,
                "valid_run": not spec.official_adapter,
                "beats_zero_filled": test["psnr"] > ZERO_FILLED["psnr"],
                "config": payload["config"],
                "params": model["params"],
                "train_sec": model["train_sec"],
                "best_val": best_val,
                "test": test,
                "history": model["history"],
            }
        )
    return runs


def ordered_runs(runs, valid_only: bool = False):
    selected = (
        [run for run in runs if run["valid_run"] and run["beats_zero_filled"]]
        if valid_only
        else runs
    )
    return sorted(selected, key=lambda run: run["test"]["psnr"], reverse=True)


def write_summary_csv(runs, out_path: Path):
    ranking = ordered_runs(runs)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank_all",
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
        for rank, run in enumerate(ranking, start=1):
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
                    f"{run['train_sec'] / 3600:.3f}",
                    run["config"]["mask_key"],
                    run["config"]["seed"],
                    run["config"]["epochs"],
                ]
            )


def markdown_ranking(runs):
    lines = [
        "| 总排名 | 模型 | 有效运行 | 高于零填充 | 参数量 | 测试 PSNR | SSIM | NMSE |",
        "|---:|---|---|---|---:|---:|---:|---:|",
        (
            f"| - | Zero-filled | 基线 | - | - | "
            f"{ZERO_FILLED['psnr']:.3f} | {ZERO_FILLED['ssim']:.3f} | "
            f"{ZERO_FILLED['nmse']:.4f} |"
        ),
    ]
    for rank, run in enumerate(ordered_runs(runs), start=1):
        test = run["test"]
        lines.append(
            f"| {rank} | {run['label']} | "
            f"{'是' if run['valid_run'] else '否'} | "
            f"{'是' if run['beats_zero_filled'] else '否'} | "
            f"{run['params']:,} | {test['psnr']:.3f} | "
            f"{test['ssim']:.4f} | {test['nmse']:.4f} |"
        )
    return "\n".join(lines)


def plot_ranked_metrics(runs, out_path: Path):
    ranking = ordered_runs(runs)
    labels = [run["label"] for run in ranking]
    y = np.arange(len(ranking))
    metrics = [
        ("psnr", "Test PSNR (dB)", ZERO_FILLED["psnr"], "{:.2f}"),
        ("ssim", "Test SSIM", ZERO_FILLED["ssim"], "{:.3f}"),
        ("nmse", "Test NMSE", ZERO_FILLED["nmse"], "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), constrained_layout=True)
    for ax, (key, title, baseline, value_format) in zip(axes, metrics):
        values = [run["test"][key] for run in ranking]
        bars = ax.barh(
            y,
            values,
            color=[run["color"] for run in ranking],
            alpha=0.88,
        )
        for bar, run in zip(bars, ranking):
            if run["official_adapter"]:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
            if not run["valid_run"] or not run["beats_zero_filled"]:
                bar.set_edgecolor("#7f1d1d")
                bar.set_linewidth(1.2)
        ax.axvline(
            baseline,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label="Zero-filled",
        )
        ax.text(
            baseline,
            -1.15,
            f"zero-filled {value_format.format(baseline)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.22)
        for yi, value in zip(y, values):
            ax.text(
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
        Patch(color="#9467bd", label="Direct U-Net"),
        Patch(color="#d62728", label="Official adapter"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Invalid run"),
        Line2D([0], [0], color="#222222", linestyle="--", label="Zero-filled"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=8)
    fig.suptitle("M4Raw medium model-zoo test metrics", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_zero_filled_gain(runs, out_path: Path):
    ranking = ordered_runs(runs)
    labels = [run["label"] for run in ranking]
    gains = [run["test"]["psnr"] - ZERO_FILLED["psnr"] for run in ranking]
    y = np.arange(len(ranking))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    bars = ax.barh(
        y,
        gains,
        color=[run["color"] for run in ranking],
        alpha=0.88,
    )
    for bar, run in zip(bars, ranking):
        if run["official_adapter"]:
            bar.set_hatch("///")
            bar.set_edgecolor("black")
    ax.axvline(0.0, color="#222222", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Test PSNR gain over zero-filled ({ZERO_FILLED['psnr']:.2f} dB)")
    ax.set_title("Gain over zero-filled reconstruction")
    ax.grid(axis="x", alpha=0.22)
    for yi, gain in zip(y, gains):
        offset = 0.06 if gain >= 0 else -0.06
        ax.text(
            gain + offset,
            yi,
            f"{gain:+.2f}",
            va="center",
            ha="left" if gain >= 0 else "right",
            fontsize=7,
        )
    ax.set_xlim(min(gains) - 0.65, max(gains) + 0.65)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_and_time(runs, out_path: Path):
    internal = [run for run in runs if not run["official_adapter"]]
    official = [run for run in runs if run["official_adapter"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax = axes[0]
    for run in internal:
        ax.errorbar(
            run["params"] / 1e6,
            run["test"]["psnr"],
            yerr=run["test"]["psnr_std"],
            fmt="o",
            color=run["color"],
            markeredgecolor="black",
            markersize=6,
            capsize=2,
            alpha=0.9,
        )
    for run in official:
        ax.errorbar(
            run["params"] / 1e6,
            run["test"]["psnr"],
            yerr=run["test"]["psnr_std"],
            fmt="s",
            color=run["color"],
            markeredgecolor="black",
            markersize=7,
            capsize=2,
        )
    ax.axhline(ZERO_FILLED["psnr"], color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("Parameter efficiency")
    ax.grid(alpha=0.22)
    annotations = {
        "VarNet non-shared CNN": (6, 7, "left"),
        "VarNet shared CNN": (-6, 8, "right"),
        "AENet recurrent CNN": (0, -28, "center"),
        "VarNet non-shared U-Net": (6, 7, "left"),
        "Official E2E VarNet": (-6, -14, "right"),
        "Official HUMUS-Net": (6, -14, "left"),
    }
    for run in runs:
        if run["label"] not in annotations:
            continue
        dx, dy, alignment = annotations[run["label"]]
        ax.annotate(
            run["label"],
            (run["params"] / 1e6, run["test"]["psnr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=alignment,
            fontsize=7,
        )
    ax.text(
        0.03,
        0.78,
        "19 internal runs above zero-filled:\n30.41 to 30.80 dB",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    ax = axes[1]
    for run in internal:
        ax.scatter(
            run["train_sec"] / 3600,
            run["test"]["psnr"],
            color=run["color"],
            edgecolor="black",
            s=45,
            alpha=0.9,
        )
    for run in official:
        ax.scatter(
            run["train_sec"] / 3600,
            run["test"]["psnr"],
            color=run["color"],
            edgecolor="black",
            marker="s",
            s=55,
        )
    ax.axhline(ZERO_FILLED["psnr"], color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Wall-clock training time (hours)")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("Quality versus training cost")
    ax.grid(alpha=0.22)
    time_annotations = {
        "VarNet non-shared CNN": (6, 7, "left"),
        "Official E2E VarNet": (6, -13, "left"),
        "Official HUMUS-Net": (-6, -13, "right"),
    }
    for run in runs:
        if run["label"] not in time_annotations:
            continue
        dx, dy, alignment = time_annotations[run["label"]]
        ax.annotate(
            run["label"],
            (run["train_sec"] / 3600, run["test"]["psnr"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=alignment,
            fontsize=7,
        )
    ax.text(
        0.03,
        0.78,
        "19 internal runs above zero-filled:\n30.41 to 30.80 dB",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    fig.suptitle("Efficiency of the M4Raw medium model zoo", fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selected_curves(runs, out_path: Path):
    selected = [
        "VarNet non-shared CNN",
        "VarNet shared CNN",
        "AENet recurrent CNN",
        "AENet recurrent U-Net",
        "Official E2E VarNet",
        "Official HUMUS-Net",
    ]
    lookup = {run["label"]: run for run in runs}
    fig, ax = plt.subplots(figsize=(11, 6))
    for label in selected:
        run = lookup[label]
        epochs = [row["epoch"] for row in run["history"]]
        validation = [row["psnr"] for row in run["history"]]
        ax.plot(
            epochs,
            validation,
            label=label,
            color=run["color"],
            linestyle=run["linestyle"],
            linewidth=1.9,
        )
        ax.scatter(
            [run["best_val"]["epoch"]],
            [run["best_val"]["psnr"]],
            color=run["color"],
            edgecolor="black",
            s=28,
            zorder=5,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("Validation trajectories: competitive models and official adapters")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_report(runs):
    ranking = ordered_runs(runs)
    valid = ordered_runs(runs, valid_only=True)
    by_label = {run["label"]: run for run in runs}
    best = valid[0]
    best_aenet = next(run for run in valid if run["family"] == "AENet")
    best_shared = by_label["VarNet shared CNN"]
    best_nonshared = by_label["VarNet non-shared CNN"]
    aenet_base = by_label["AENet CNN"]
    official_varnet = by_label["Official E2E VarNet"]
    official_humus = by_label["Official HUMUS-Net"]
    ista = by_label["ISTA-style"]
    unshar_unet = by_label["VarNet non-shared U-Net"]
    shared_unet = by_label["VarNet shared U-Net"]

    data_paths = {run["config"]["data"] for run in runs}
    masks = {run["config"]["mask_key"] for run in runs}
    seeds = {run["config"]["seed"] for run in runs}
    epochs = {run["config"]["epochs"] for run in runs}

    official_gap_humus = official_humus["test"]["psnr"] - ZERO_FILLED["psnr"]
    official_gap_varnet = official_varnet["test"]["psnr"] - ZERO_FILLED["psnr"]
    aenet_gap_shared = (
        best_aenet["test"]["psnr"] - best_shared["test"]["psnr"]
    )
    aenet_gap_nonshared = (
        best_aenet["test"]["psnr"] - best_nonshared["test"]["psnr"]
    )
    recurrence_gain = (
        best_aenet["test"]["psnr"] - aenet_base["test"]["psnr"]
    )
    param_ratio = best_nonshared["params"] / best_shared["params"]
    psnr_gain_nonshared = (
        best_nonshared["test"]["psnr"] - best_shared["test"]["psnr"]
    )

    return f"""# M4Raw Medium Model-Zoo Results v7

## 分析范围

- 共解析 {len(runs)} 个已完成的中规模 JSON 结果。
- 所有文件使用同一数据路径、R=4 掩码、seed 42 和 100 epochs。
- 协议一致性检查：data={len(data_paths)}, mask={sorted(masks)}, seed={sorted(seeds)}, epochs={sorted(epochs)}。
- 中规模测试集为 200 个 slice，指标为 slice-level mean。
- 零填充参考值采用当前论文中相同的 medium protocol：PSNR {ZERO_FILLED['psnr']:.2f} dB，SSIM {ZERO_FILLED['ssim']:.3f}，NMSE {ZERO_FILLED['nmse']:.4f}。
- 两个官方适配器运行单独标记为无效运行，不进入有效模型榜首判断。

## 完整排名

{markdown_ranking(runs)}

## 关键结论

1. 当前有效模型第一名是 **{best['label']}**，测试结果为 PSNR {best['test']['psnr']:.3f} dB、SSIM {best['test']['ssim']:.4f}、NMSE {best['test']['nmse']:.4f}，比零填充高 {best['test']['psnr'] - ZERO_FILLED['psnr']:+.3f} dB。
2. 最佳 AENet 是 **{best_aenet['label']}**，测试结果为 PSNR {best_aenet['test']['psnr']:.3f} dB、SSIM {best_aenet['test']['ssim']:.4f}、NMSE {best_aenet['test']['nmse']:.4f}。
3. 最佳 AENet 与共享 CNN VarNet 的差距为 {aenet_gap_shared:+.4f} dB，数值上基本持平；与参数更多一档的非共享 CNN VarNet 相差 {aenet_gap_nonshared:+.4f} dB。JSON 只有 slice-level 汇总和标准差，没有成对样本，因此不能据此声称统计显著或等价。
4. 循环状态把 AENet CNN 提升到 {best_aenet['test']['psnr']:.3f} dB，比基础 AENet CNN 高 {recurrence_gain:+.3f} dB，这是当前最明确的 AENet 内部增益。
5. 非共享 CNN VarNet 使用了共享 CNN VarNet 约 {param_ratio:.1f} 倍的参数，只换来 {psnr_gain_nonshared:+.3f} dB。参数规模不是主要瓶颈。
6. 更大的 U-Net 先验没有带来优势：非共享 U-Net 为 {unshar_unet['test']['psnr']:.3f} dB，共享 U-Net 为 {shared_unet['test']['psnr']:.3f} dB，均低于 0.15M 参数的共享 CNN VarNet。
7. 经典展开比较组中，ADMM、LPD 和 FISTA 集中在约 30.41 到 30.42 dB，MoDL 为 {by_label['MoDL-style']['test']['psnr']:.3f} dB。ISTA-style 仅 {ista['test']['psnr']:.3f} dB，低于零填充 {ista['test']['psnr'] - ZERO_FILLED['psnr']:+.3f} dB。

## 新加入的官方模型

- **Official HUMUS-Net**：{official_humus['params']:,} 参数，训练 {official_humus['train_sec'] / 3600:.2f} 小时，测试 PSNR {official_humus['test']['psnr']:.3f} dB，相对零填充 {official_gap_humus:+.3f} dB。
- **Official E2E VarNet**：{official_varnet['params']:,} 参数，训练 {official_varnet['train_sec'] / 3600:.2f} 小时，测试 PSNR {official_varnet['test']['psnr']:.3f} dB，相对零填充 {official_gap_varnet:+.3f} dB。
- 两者验证曲线在约 22 dB 附近长期平台化，说明问题更可能来自官方模型适配、数据预处理、灵敏度图接口或训练协议不一致，而不是“官方方法理论上只能达到这个水平”。
- 共同适配点是两个 adapter 都丢弃了本地 `smap`，只把 masked k-space 和 mask 传给官方模型，并由官方模型内部重新估计灵敏度图。所有内部基线则直接使用 prepared M4Raw sensitivity maps。两种官方模型在同一路径上同时停在约 22 dB，因此应优先核对这条共有接口的坐标、中心频率带、灵敏度估计和归一化约定。
- 在修正适配前，不建议把这两个数字写入论文作为官方方法性能。

## 展示文件

- `figures/figure-01-ranked-test-metrics.png`：PSNR、SSIM、NMSE 完整排名。
- `figures/figure-02-zero-filled-gain.png`：相对零填充的 PSNR 增益。
- `figures/figure-03-efficiency.png`：参数效率与训练成本。
- `figures/figure-04-validation-trajectories.png`：竞争模型和官方适配器的验证轨迹。
- `summary.csv`：用于后续制表的完整机器可读结果。

## 证据边界

- 当前只有 seed 42 和单个 R=4 掩码，不能进行跨种子稳健性结论。
- 当前 JSON 不包含逐 slice 预测或成对统计量，不能运行配对检验。
- 训练与测试划分相同，但不同实现可能仍有内部训练细节差异。
- 官方适配器异常是运行有效性结论，不是对官方论文方法的否定。
"""


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "analysis-output-v7"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(script_dir)
    write_summary_csv(runs, out_dir / "summary.csv")
    (out_dir / "analysis-report.md").write_text(
        format_report(runs), encoding="utf-8"
    )
    plot_ranked_metrics(runs, figures_dir / "figure-01-ranked-test-metrics.png")
    plot_zero_filled_gain(runs, figures_dir / "figure-02-zero-filled-gain.png")
    plot_parameter_and_time(runs, figures_dir / "figure-03-efficiency.png")
    plot_selected_curves(
        runs, figures_dir / "figure-04-validation-trajectories.png"
    )

    print(f"Wrote analysis to {out_dir}")
    print(f"Completed valid runs: {len(ordered_runs(runs, valid_only=True))}")
    best = ordered_runs(runs, valid_only=True)[0]
    print(
        f"Best valid run: {best['label']} "
        f"PSNR={best['test']['psnr']:.4f} "
        f"SSIM={best['test']['ssim']:.4f} "
        f"NMSE={best['test']['nmse']:.4f}"
    )


if __name__ == "__main__":
    main()
