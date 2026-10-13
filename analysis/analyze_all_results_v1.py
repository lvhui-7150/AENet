"""Build a consolidated analysis bundle for every result JSON in resute."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


M4RAW_SUMMARY = Path("analysis-output-v7") / "summary.csv"
KNEE_SUMMARY = Path("analysis-fastmri-knee-v1") / "summary.csv"
SMOKE_FILES = (
    "m4raw_results_hpc_smoke.json",
    "m4raw_results_hpc_gpu_smoke.json",
)
SMOKE_ZERO_FILLED = {
    "psnr": 27.11,
    "ssim": 0.659,
    "nmse": 0.0553,
}
SMOKE_LABELS = {
    "aenet": "AENet CNN",
    "varnet_shared": "VarNet shared CNN",
    "varnet": "VarNet non-shared CNN",
}


def parse_float(value):
    if value in (None, ""):
        return None
    return float(value)


def load_medium_summary(path: Path, dataset: str):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw["model"] == "Zero-filled":
                continue
            rank_key = "rank_all" if "rank_all" in raw else "rank"
            valid_key = "valid" if "valid" in raw else "valid_run"
            rows.append(
                {
                    "dataset": dataset,
                    "source": str(path),
                    "device": "",
                    "label": raw["model"],
                    "params": int(raw["params"]),
                    "epochs": int(raw.get("epochs", 0) or 0),
                    "best_epoch": int(raw["best_epoch"]),
                    "psnr": parse_float(raw["test_psnr"]),
                    "psnr_std": parse_float(raw.get("test_psnr_std")),
                    "ssim": parse_float(raw["test_ssim"]),
                    "nmse": parse_float(raw["test_nmse"]),
                    "hours": parse_float(raw.get("train_hours")),
                    "valid": bool(int(raw[valid_key])),
                    "beats_zero": bool(int(raw["beats_zero_filled"])),
                    "rank_all": int(raw[rank_key]),
                }
            )
    for row in rows:
        row["rank_valid"] = None
    valid = sorted(
        (row for row in rows if row["valid"]),
        key=lambda row: row["psnr"],
        reverse=True,
    )
    for rank, row in enumerate(valid, start=1):
        row["rank_valid"] = rank
    top_psnr = valid[0]["psnr"]
    for row in rows:
        row["gap_to_top"] = top_psnr - row["psnr"]
    return rows


def load_smoke_runs(result_dir: Path):
    rows = []
    for filename in SMOKE_FILES:
        path = result_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model_name, model in payload["models"].items():
            test = model["test"]
            rows.append(
                {
                    "dataset": "M4Raw smoke",
                    "source": filename,
                    "device": payload["device"],
                    "label": SMOKE_LABELS.get(model_name, model_name),
                    "params": model["params"],
                    "epochs": payload["config"]["epochs"],
                    "best_epoch": model["best_val"]["epoch"],
                    "psnr": test["psnr"],
                    "psnr_std": test["psnr_std"],
                    "ssim": test["ssim"],
                    "nmse": test["nmse"],
                    "hours": model["train_sec"] / 3600,
                    "valid": True,
                    "beats_zero": test["psnr"] > SMOKE_ZERO_FILLED["psnr"],
                    "rank_all": None,
                    "rank_valid": None,
                    "gap_to_top": None,
                }
            )
    return rows


def write_all_runs_csv(rows, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "source",
                "device",
                "model",
                "params",
                "epochs",
                "best_epoch",
                "test_psnr",
                "test_psnr_std",
                "test_ssim",
                "test_nmse",
                "train_hours",
                "valid_run",
                "beats_zero_filled",
                "rank_all",
                "rank_valid",
                "gap_to_dataset_top",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["source"],
                    row["device"],
                    row["label"],
                    row["params"],
                    row["epochs"],
                    row["best_epoch"],
                    f"{row['psnr']:.6f}",
                    (
                        f"{row['psnr_std']:.6f}"
                        if row["psnr_std"] is not None
                        else ""
                    ),
                    f"{row['ssim']:.6f}",
                    f"{row['nmse']:.6f}",
                    f"{row['hours']:.6f}",
                    int(row["valid"]),
                    int(row["beats_zero"]),
                    "" if row["rank_all"] is None else row["rank_all"],
                    "" if row["rank_valid"] is None else row["rank_valid"],
                    (
                        f"{row['gap_to_top']:.6f}"
                        if row["gap_to_top"] is not None
                        else ""
                    ),
                ]
            )


def build_common_table(m4_rows, knee_rows):
    m4 = {row["label"]: row for row in m4_rows}
    knee = {row["label"]: row for row in knee_rows}
    common = []
    for label in sorted(set(m4) & set(knee)):
        first = m4[label]
        second = knee[label]
        common.append(
            {
                "label": label,
                "m4_psnr": first["psnr"],
                "m4_rank": first["rank_valid"],
                "m4_gap": first["gap_to_top"],
                "m4_valid": first["valid"],
                "knee_psnr": second["psnr"],
                "knee_rank": second["rank_valid"],
                "knee_gap": second["gap_to_top"],
                "knee_valid": second["valid"],
            }
        )
    valid_common = [
        row for row in common if row["m4_valid"] and row["knee_valid"]
    ]
    valid_common.sort(
        key=lambda row: (row["m4_rank"] + row["knee_rank"]) / 2.0
    )
    invalid_common = sorted(
        (row for row in common if not (row["m4_valid"] and row["knee_valid"])),
        key=lambda row: row["label"],
    )
    return valid_common + invalid_common


def write_common_csv(common, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "m4raw_psnr",
                "m4raw_valid_rank",
                "m4raw_gap_to_top",
                "m4raw_valid_run",
                "knee_psnr",
                "knee_valid_rank",
                "knee_gap_to_top",
                "knee_valid_run",
            ]
        )
        for row in common:
            writer.writerow(
                [
                    row["label"],
                    f"{row['m4_psnr']:.6f}",
                    "" if row["m4_rank"] is None else row["m4_rank"],
                    f"{row['m4_gap']:.6f}",
                    int(row["m4_valid"]),
                    f"{row['knee_psnr']:.6f}",
                    "" if row["knee_rank"] is None else row["knee_rank"],
                    f"{row['knee_gap']:.6f}",
                    int(row["knee_valid"]),
                ]
            )


def markdown_smoke_table(smoke):
    lines = [
        "| 模型 | 设备 | 参数量 | Epochs | 测试 PSNR | SSIM | NMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(smoke, key=lambda item: (item["label"], item["device"])):
        lines.append(
            f"| {row['label']} | {row['device']} | {row['params']:,} | "
            f"{row['epochs']} | {row['psnr']:.3f} | {row['ssim']:.4f} | "
            f"{row['nmse']:.4f} |"
        )
    return "\n".join(lines)


def markdown_common_table(common):
    def fmt(value, digits):
        if value is None:
            return "-"
        try:
            if math.isnan(value):
                return "-"
        except TypeError:
            pass
        return f"{value:.{digits}f}"

    lines = [
        "| 模型 | M4Raw PSNR | M4Raw 有效排名 | M4Raw 距榜首 | Knee PSNR | Knee 有效排名 | Knee 距榜首 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in common:
        m4_rank = "-" if row["m4_rank"] is None else str(row["m4_rank"])
        knee_rank = "-" if row["knee_rank"] is None else str(row["knee_rank"])
        lines.append(
            f"| {row['label']} | {fmt(row['m4_psnr'], 3)} | {m4_rank} | "
            f"{fmt(row['m4_gap'], 3)} | {fmt(row['knee_psnr'], 3)} | "
            f"{knee_rank} | {fmt(row['knee_gap'], 3)} |"
        )
    return "\n".join(lines)


def plot_rank_matrix(common, out_path: Path):
    rows = [
        row
        for row in common
        if row["m4_valid"]
        and row["knee_valid"]
        and row["m4_rank"] is not None
        and row["knee_rank"] is not None
    ]
    matrix = np.asarray(
        [[row["m4_rank"], row["knee_rank"]] for row in rows],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(6.8, max(7.0, 0.42 * len(rows))))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto", vmin=1)
    axis.set_xticks([0, 1], ["M4Raw medium", "fastMRI Knee medium"])
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels([row["label"] for row in rows], fontsize=8)
    axis.set_title("Valid-model rank across completed datasets")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rank = int(matrix[i, j])
            color = "white" if rank <= max(3, matrix.max() * 0.45) else "black"
            axis.text(j, i, str(rank), ha="center", va="center", color=color)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Rank among valid runs (1 is best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gap_to_top(common, out_path: Path):
    rows = [
        row
        for row in common
        if row["m4_valid"]
        and row["knee_valid"]
        and row["m4_gap"] is not None
        and row["knee_gap"] is not None
    ]
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10.5, max(7.0, 0.45 * len(rows))))
    axis.barh(
        y - 0.19,
        [row["m4_gap"] for row in rows],
        height=0.36,
        color="#1f77b4",
        label="M4Raw medium",
    )
    axis.barh(
        y + 0.19,
        [row["knee_gap"] for row in rows],
        height=0.36,
        color="#ff7f0e",
        label="fastMRI Knee medium",
    )
    axis.set_yticks(y)
    axis.set_yticklabels([row["label"] for row in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("PSNR gap to the dataset's best valid model (dB, lower is better)")
    axis.set_title("Within-dataset comparison of common architectures")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_rows(rows, count=5):
    valid = sorted(
        (row for row in rows if row["valid"]),
        key=lambda row: row["psnr"],
        reverse=True,
    )
    return valid[:count]


def best_label(rows, predicate):
    candidates = [row for row in rows if row["valid"] and predicate(row)]
    return max(candidates, key=lambda row: row["psnr"])


def smoke_reproducibility(smoke):
    by_model = {}
    for row in smoke:
        by_model.setdefault(row["label"], {})[row["device"]] = row
    lines = [
        "| 模型 | CPU/GPU PSNR 差 | SSIM 差 | NMSE 差 |",
        "|---|---:|---:|---:|",
    ]
    for label, devices in sorted(by_model.items()):
        cpu = devices.get("cpu")
        gpu = devices.get("cuda")
        if cpu is None or gpu is None:
            continue
        lines.append(
            f"| {label} | {abs(cpu['psnr'] - gpu['psnr']):.6f} | "
            f"{abs(cpu['ssim'] - gpu['ssim']):.6f} | "
            f"{abs(cpu['nmse'] - gpu['nmse']):.6f} |"
        )
    return "\n".join(lines)


def format_report(m4_rows, knee_rows, smoke_rows, common):
    m4_top = top_rows(m4_rows)
    knee_top = top_rows(knee_rows)
    m4_best_aenet = best_label(
        m4_rows, lambda row: row["label"].startswith("AENet")
    )
    knee_best_aenet = best_label(
        knee_rows, lambda row: row["label"].startswith("AENet")
    )
    m4_recurrent_cnn = next(
        row for row in m4_rows if row["label"] == "AENet recurrent CNN"
    )
    knee_recurrent_cnn = next(
        row for row in knee_rows if row["label"] == "AENet recurrent CNN"
    )
    m4_official = {
        row["label"]: row
        for row in m4_rows
        if row["label"].startswith("Official")
    }
    knee_official = {
        row["label"]: row
        for row in knee_rows
        if row["label"].startswith("Official")
    }
    smoke_delta = 0.0
    by_model = {}
    for row in smoke_rows:
        by_model.setdefault(row["label"], {})[row["device"]] = row
    for devices in by_model.values():
        if "cpu" in devices and "cuda" in devices:
            smoke_delta = max(
                smoke_delta,
                abs(devices["cpu"]["psnr"] - devices["cuda"]["psnr"]),
                abs(devices["cpu"]["ssim"] - devices["cuda"]["ssim"]),
                abs(devices["cpu"]["nmse"] - devices["cuda"]["nmse"]),
            )

    return f"""# All Results Consolidated Analysis

## Scope

| 实验组 | JSON 文件 | 运行记录 | Epochs | 当前状态 | 分析用途 |
|---|---:|---:|---:|---|---|
| M4Raw smoke | 2 | 6 | 10 | CPU/GPU 对照完成 | 工程正确性与复现性检查 |
| M4Raw medium | 22 | 22 | 100 | 22/22 完成 | 多线圈正式开发排名 |
| fastMRI Knee medium | 20 | 20 | 100 | 20/22 完成 | 单线圈开发排名 |
| 合计 | 44 | 48 | - | - | 分层分析，不混合绝对 PSNR |

## M4Raw Smoke

Smoke 数据为小规模 10-epoch 工程验证，不应与 medium 结果放在同一排行榜。
当前论文使用的零填充参考为 PSNR {SMOKE_ZERO_FILLED['psnr']:.3f}、
SSIM {SMOKE_ZERO_FILLED['ssim']:.3f}、
NMSE {SMOKE_ZERO_FILLED['nmse']:.4f}。

{markdown_smoke_table(smoke_rows)}

CPU/GPU 最大指标差异为 {smoke_delta:.6g}，说明该 smoke 训练路径在两种设备上
具有数值一致性。该结论仅证明工程实现一致，不证明性能领先。

## M4Raw Medium

完整排名保存在 `analysis-output-v7/summary.csv`。前五名为：

{markdown_common_table([
    {
        "label": row["label"],
        "m4_psnr": row["psnr"],
        "m4_rank": row["rank_valid"],
        "m4_gap": row["gap_to_top"],
        "knee_psnr": float("nan"),
        "knee_rank": None,
        "knee_gap": float("nan"),
    }
    for row in m4_top
])}

- 最佳有效模型：**{m4_top[0]['label']}**，PSNR
  {m4_top[0]['psnr']:.3f}，SSIM {m4_top[0]['ssim']:.4f}，
  NMSE {m4_top[0]['nmse']:.4f}。
- 最佳 AENet：**{m4_best_aenet['label']}**，PSNR
  {m4_best_aenet['psnr']:.3f}，与榜首相差
  {m4_best_aenet['gap_to_top']:.3f} dB。
- AENet recurrent CNN 为 {m4_recurrent_cnn['psnr']:.3f} dB，
  有效排名 {m4_recurrent_cnn['rank_valid']}。
- Official E2E VarNet 和 Official HUMUS-Net 均低于零填充，标记为无效适配运行：
  E2E VarNet {m4_official['Official E2E VarNet']['psnr']:.3f} dB，
  HUMUS-Net {m4_official['Official HUMUS-Net']['psnr']:.3f} dB。

## fastMRI Knee Medium

完整排名保存在 `analysis-fastmri-knee-v1/summary.csv`。前五名为：

{markdown_common_table([
    {
        "label": row["label"],
        "m4_psnr": float("nan"),
        "m4_rank": None,
        "m4_gap": float("nan"),
        "knee_psnr": row["psnr"],
        "knee_rank": row["rank_valid"],
        "knee_gap": row["gap_to_top"],
    }
    for row in knee_top
])}

- 最佳有效模型：**{knee_top[0]['label']}**，PSNR
  {knee_top[0]['psnr']:.3f}，SSIM {knee_top[0]['ssim']:.4f}，
  NMSE {knee_top[0]['nmse']:.4f}。
- 最佳 AENet：**{knee_best_aenet['label']}**，PSNR
  {knee_best_aenet['psnr']:.3f}，与榜首相差
  {knee_best_aenet['gap_to_top']:.3f} dB。
- AENet recurrent CNN 为 {knee_recurrent_cnn['psnr']:.3f} dB，
  有效排名 {knee_recurrent_cnn['rank_valid']}。
- 两个 official adapters 仍低于零填充：Official E2E VarNet
  {knee_official['Official E2E VarNet']['psnr']:.3f} dB，Official HUMUS-Net
  {knee_official['Official HUMUS-Net']['psnr']:.3f} dB。

## Common Architecture Comparison

下表只比较两个 medium 数据集中都存在的模型。由于两套数据不是同一任务，
比较重点是有效排名和距各自榜首的差距，而不是绝对 PSNR。

{markdown_common_table(common)}

## Main Conclusions

1. M4Raw 多线圈数据上，最佳有效模型是非共享 CNN VarNet
   ({m4_top[0]['psnr']:.3f} dB)，AENet recurrent CNN 仅落后
   {m4_recurrent_cnn['gap_to_top']:.3f} dB。
2. Knee 单线圈数据上，最佳有效模型是非共享 U-Net VarNet
   ({knee_top[0]['psnr']:.3f} dB)，AENet recurrent U-Net 仅落后
   {knee_best_aenet['gap_to_top']:.3f} dB。
3. AENet recurrent 系列是最值得继续扩展的 AENet 方向：
   recurrent U-Net 在 M4Raw/Knee 分别为有效第 7/2 名；
   recurrent CNN 在 M4Raw/Knee 分别为第
   {m4_recurrent_cnn['rank_valid']}/{knee_recurrent_cnn['rank_valid']} 名，
   且参数量约 0.17M。
4. Feature-image、FI attention 和 CSME 在两个数据集中都没有稳定进入榜首，
   当前不支持“更复杂模块一定更好”。
5. Direct U-Net 和 Attention U-Net 在两个数据集上都明显落后于展开模型，
   说明数据一致性展开是主要收益来源。
6. 两个 official adapters 在 M4Raw 和 Knee 上都失败，问题具有跨数据集共性，
   应优先定位共同适配层，而不是继续扩大训练。
7. Knee 上 FISTA/LPD 很强，而 M4Raw 上是 VarNet/AENet 领先。
   两次数据协议的线圈数、数据规模和 k-space 来源不同，
   因此不能推出统一的方法排序。

## Statistical Status

- Medium 结果均只有一个 seed 和一个 mask。
- JSON 只保存跨 slice 的均值和标准差，没有逐 slice paired values。
- 因此不能计算可靠 p-value，也不能声称某一模型显著胜出。
- Smoke CPU/GPU 对照只支持数值复现性，不支持性能显著性。
- 最终统计需要逐 slice 指标、volume 映射、多个 seed 和多个 mask。

## Claim Candidates

- Claim: AENet recurrent 系列是当前最稳定的 AENet 扩展方向。
  - Source evidence: recurrent U-Net 在两个协议中排名 2 和 7；
    recurrent CNN 排名 3 和 9，且参数量约 0.17M。
  - Allowed wording: “在两个当前开发协议中，recurrent 系列是优先扩展方向”。
  - Forbidden stronger wording: “recurrent CNN 在两个数据集上都优于其他模型”。
  - Uncertainty: 单 seed、单 mask、缺少 paired metrics。
  - Decision: keep.
- Claim: 两个 official adapters 当前无效。
  - Source evidence: 在 M4Raw 和 Knee 上均低于零填充。
  - Allowed wording: “当前适配运行无效”。
  - Forbidden stronger wording: “官方方法本身性能差”。
  - Uncertainty: 具体接口故障尚未定位。
  - Decision: keep.
- Claim: 复杂 feature 模块没有稳定收益。
  - Source evidence: Feature-image、FI attention、CSME 在两个协议中均未领先。
  - Allowed wording: “当前证据不支持其稳定增益”。
  - Forbidden stronger wording: “这些模块无效”。
  - Uncertainty: 数据和训练预算有限。
  - Decision: weaken.
"""


def format_stats_appendix(smoke_rows):
    return f"""# All Results Statistics Appendix

## Evidence Inventory

| 实验组 | 运行数 | Seed | Mask | 统计状态 |
|---|---:|---:|---:|---|
| M4Raw smoke | 6 | 1 | 1 | 仅工程复现性 |
| M4Raw medium | 22 | 1 | 1 | 描述统计，无显著性 |
| fastMRI Knee medium | 20 | 1 | 1 | 描述统计，无显著性 |

## Smoke Reproducibility

{smoke_reproducibility(smoke_rows)}

## Inferential Blockers

- 没有逐 slice PSNR、SSIM、NMSE。
- 没有 slice 到 volume 的映射。
- 没有 seed-level repeats。
- 没有多个 mask 的重复实验。
- 因此不能执行 paired t-test、Wilcoxon、clustered bootstrap 或可靠置信区间。

## Unit-Of-Analysis Warning

即使补齐逐 slice 指标，也不能默认把同一 volume 的所有 slice 视为独立样本。
后续统计应至少同时报告 slice-level paired test 和 volume-level aggregation，
或使用按 volume 聚类的 bootstrap。
"""


def format_figure_catalog():
    return """# All Results Figure Catalog

## Group-Specific Figures

- `analysis-output-v7/figures/`：M4Raw medium 排名、零填充增益、参数效率和轨迹。
- `analysis-fastmri-knee-v1/figures/`：Knee medium 排名、零填充增益、参数效率和轨迹。

## analysis-all-results-v1/figures/figure-01-cross-dataset-rank-matrix.png

- Purpose: 比较共同模型在两个 medium 数据集中的有效排名。
- Data source: 两个 summary.csv 的有效模型排名。
- What to notice: AENet recurrent 系列整体位于 AENet 组的前部；
  FISTA 在 Knee 上明显优于其在 M4Raw 上的位置。
- Decision impact: 不以单一数据集的名次宣布架构优劣。

## analysis-all-results-v1/figures/figure-02-gap-to-top.png

- Purpose: 在各自数据集内部比较模型距榜首的 PSNR 差距。
- Data source: 两个 medium 数据集的 test PSNR。
- What to notice: 多个模型集中在榜首附近，
  领先幅度通常小于跨 slice 标准差。
- Decision impact: 后续应优先补齐重复实验和成对统计。
"""


def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    out_dir = project_dir / "analysis-all-results-v1"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    m4_rows = load_medium_summary(project_dir / M4RAW_SUMMARY, "M4Raw medium")
    knee_rows = load_medium_summary(
        project_dir / KNEE_SUMMARY, "fastMRI Knee medium"
    )
    smoke_rows = load_smoke_runs(script_dir)
    common = build_common_table(m4_rows, knee_rows)

    write_all_runs_csv(
        smoke_rows + m4_rows + knee_rows, out_dir / "all-runs.csv"
    )
    write_common_csv(common, out_dir / "common-models.csv")
    (out_dir / "analysis-report.md").write_text(
        format_report(m4_rows, knee_rows, smoke_rows, common),
        encoding="utf-8",
    )
    (out_dir / "stats-appendix.md").write_text(
        format_stats_appendix(smoke_rows),
        encoding="utf-8",
    )
    (out_dir / "figure-catalog.md").write_text(
        format_figure_catalog(),
        encoding="utf-8",
    )
    plot_rank_matrix(
        common,
        figures_dir / "figure-01-cross-dataset-rank-matrix.png",
    )
    plot_gap_to_top(
        common,
        figures_dir / "figure-02-gap-to-top.png",
    )

    print(f"Wrote consolidated analysis to {out_dir}")
    print(
        f"Loaded runs: smoke={len(smoke_rows)}, "
        f"m4raw_medium={len(m4_rows)}, knee_medium={len(knee_rows)}"
    )
    print(f"Common models: {len(common)}")


if __name__ == "__main__":
    main()
