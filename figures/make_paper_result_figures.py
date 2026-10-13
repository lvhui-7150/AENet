"""Create publication-ready model-zoo figures for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_light": "#D88F8A",
    "neutral": "#7A7A7A",
    "neutral_light": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#C89B1A",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 1.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            if model == "Zero-filled" or "official" in model.lower():
                continue
            rows.append(
                {
                    "model": model,
                    "params": int(row["params"]),
                    "psnr": float(row["test_psnr"]),
                    "ssim": float(row["test_ssim"]),
                    "nmse": float(row["test_nmse"]),
                    "train_hours": float(row.get("train_hours") or "nan"),
                }
            )
    return rows


def save(fig, out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def add_normalized_metrics(rows: list[dict]) -> None:
    psnr = np.asarray([row["psnr"] for row in rows], dtype=float)
    ssim = np.asarray([row["ssim"] for row in rows], dtype=float)
    nmse = np.asarray([row["nmse"] for row in rows], dtype=float)
    params = np.asarray([row["params"] for row in rows], dtype=float)
    param_efficiency = -np.log10(params)

    norm_psnr = minmax(psnr)
    norm_ssim = minmax(ssim)
    norm_nmse = 1.0 - minmax(nmse)
    norm_params = minmax(param_efficiency)

    for idx, row in enumerate(rows):
        row["norm"] = {
            "PSNR": float(norm_psnr[idx]),
            "SSIM": float(norm_ssim[idx]),
            "NMSE": float(norm_nmse[idx]),
            "Param.\nefficiency": float(norm_params[idx]),
        }
        row["composite"] = float(
            np.mean(
                [
                    norm_psnr[idx],
                    norm_ssim[idx],
                    norm_nmse[idx],
                    norm_params[idx],
                ]
            )
        )


def metric_rank(rows: list[dict], key: str) -> dict[str, int]:
    reverse = key in {"psnr", "ssim"}
    ordered = sorted(rows, key=lambda row: row[key], reverse=reverse)
    return {row["model"]: rank + 1 for rank, row in enumerate(ordered)}


def color_for_model(model: str, fallback_index: int = 0) -> str:
    if model == "AENet v2" or model == "AENet recurrent CNN":
        return PALETTE["blue_main"]
    if model.startswith("AENet"):
        return PALETTE["blue_secondary"]
    if model.startswith("VarNet") and "non-shared" in model:
        return PALETTE["red_strong"]
    if model.startswith("VarNet"):
        return PALETTE["red_light"]
    if "FISTA" in model or "ADMM" in model or "LPD" in model or "MoDL" in model:
        return PALETTE["teal"]
    if "Feature-image" in model or "FI attention" in model:
        return PALETTE["violet"]
    if "Direct" in model or "Attention" in model:
        return PALETTE["neutral"]
    fallback = [
        PALETTE["teal"],
        PALETTE["violet"],
        PALETTE["green_3"],
        PALETTE["gold"],
        PALETTE["neutral"],
    ]
    return fallback[fallback_index % len(fallback)]


def select_top(rows: list[dict], count: int, key: str) -> list[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=True)[:count]


def select_named(rows: list[dict], names: list[str]) -> list[dict]:
    by_name = {row["model"]: row for row in rows}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"missing model rows: {missing}")
    return [by_name[name] for name in names]


def plot_metric_radar(
    rows: list[dict],
    out_dir: Path,
    stem: str,
    title: str,
    selected: list[dict],
) -> None:
    del rows
    keys = ["PSNR", "SSIM", "NMSE", "Param.\nefficiency"]
    raw = {
        "PSNR": np.asarray([row["psnr"] for row in selected], dtype=float),
        "SSIM": np.asarray([row["ssim"] for row in selected], dtype=float),
        "NMSE": 1.0 - np.asarray([row["nmse"] for row in selected], dtype=float),
        "Param.\nefficiency": -np.log10(
            np.asarray([row["params"] for row in selected], dtype=float)
        ),
    }
    normalized = {key: minmax(values) for key, values in raw.items()}
    angles = np.linspace(0.0, 2.0 * np.pi, len(keys), endpoint=False)
    angles_closed = np.append(angles, angles[0])

    fig = plt.figure(figsize=(6.1, 5.2))
    axis = fig.add_subplot(111, projection="polar")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels([])
    axis.set_xticks(angles)
    axis.set_xticklabels(keys, fontsize=9)
    axis.tick_params(axis="x", pad=17)
    axis.grid(color="#B8B8B8", linewidth=0.6, alpha=0.55)
    axis.spines["polar"].set_color("#555555")
    axis.spines["polar"].set_linewidth(0.9)

    handles = []
    labels = []
    for idx, row in enumerate(selected):
        values = np.asarray([normalized[key][idx] for key in keys])
        values_closed = np.append(values, values[0])
        color = color_for_model(row["model"], idx)
        line = axis.plot(
            angles_closed,
            values_closed,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.7,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=row["model"],
            zorder=3 + idx,
        )[0]
        is_main = row["model"] == "AENet v2" or row["model"] == "AENet recurrent CNN"
        if is_main:
            axis.fill(angles_closed, values_closed, color=color, alpha=0.11, zorder=1)
        handles.append(line)
        labels.append(row["model"])

    del title
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=8.1,
        handlelength=2.2,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.12, right=0.88, top=0.97, bottom=0.24)
    save(fig, out_dir, stem)


def plot_rank_trajectory(
    rows: list[dict],
    out_dir: Path,
    stem: str,
    title: str,
    selected_count: int = 6,
) -> None:
    ranks = {
        "PSNR": metric_rank(rows, "psnr"),
        "SSIM": metric_rank(rows, "ssim"),
        "NMSE": metric_rank(rows, "nmse"),
    }
    metric_names = list(ranks)
    average_rank = {
        row["model"]: float(np.mean([ranks[key][row["model"]] for key in metric_names]))
        for row in rows
    }
    selected = sorted(rows, key=lambda row: average_rank[row["model"]])[:selected_count]
    selected.sort(key=lambda row: average_rank[row["model"]])

    fig, axis = plt.subplots(figsize=(6.1, 5.2))
    x = np.arange(len(metric_names))
    axis.set_xlim(-0.08, len(metric_names) - 0.38)
    max_rank = max(ranks[key][row["model"]] for key in metric_names for row in selected)
    axis.set_ylim(max_rank + 0.8, 0.2)
    axis.set_xticks(x)
    axis.set_xticklabels(metric_names, fontsize=10.5, fontweight="bold")
    axis.set_ylabel("Rank among all models", fontsize=10)
    axis.set_yticks(np.arange(1, max_rank + 1))
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)

    top_n = min(3, max_rank)
    for rank in range(1, top_n + 1):
        axis.axhspan(rank - 0.5, rank + 0.5, color="#F0F4F8", alpha=0.75, zorder=0)

    endpoint_labels = []
    for idx, row in enumerate(selected):
        y = np.asarray([ranks[key][row["model"]] for key in metric_names], dtype=float)
        is_main = row["model"] == "AENet v2" or row["model"] == "AENet recurrent CNN"
        is_varnet = row["model"].startswith("VarNet")
        color = color_for_model(row["model"], idx)
        axis.plot(
            x,
            y,
            color=color,
            linewidth=2.7 if is_main else (1.9 if is_varnet else 1.25),
            alpha=1.0 if is_main or is_varnet else 0.72,
            marker="o",
            markersize=5.2 if is_main else 3.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4 if is_main else 3,
        )
        for xpos, ypos in zip(x, y):
            axis.text(
                xpos,
                ypos - 0.18,
                f"{int(ypos)}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white",
                fontweight="bold",
                zorder=5,
            )
        endpoint_labels.append((y[-1], row["model"], color, is_main))

    endpoint_labels.sort(key=lambda item: item[0])
    used_y = []
    for rank, model, color, is_main in endpoint_labels:
        label_y = rank
        while any(abs(label_y - used_y_value) < 0.32 for used_y_value in used_y):
            label_y += 0.32
        used_y.append(label_y)
        axis.plot(
            [x[-1] + 0.03, x[-1] + 0.11],
            [rank, label_y],
            color=color,
            linewidth=0.8,
            alpha=0.85,
            clip_on=False,
        )
        axis.text(
            x[-1] + 0.13,
            label_y,
            model,
            va="center",
            ha="left",
            fontsize=7.7,
            color=color,
            fontweight="bold" if is_main else "normal",
            clip_on=False,
        )

    del title
    for spine in ("right", "top"):
        axis.spines[spine].set_visible(False)
    fig.subplots_adjust(right=0.67)
    fig.tight_layout(pad=0.7)
    save(fig, out_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4raw-summary", required=True)
    parser.add_argument("--fastmri-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    apply_publication_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m4raw = load_rows(Path(args.m4raw_summary))
    fastmri = load_rows(Path(args.fastmri_summary))
    add_normalized_metrics(m4raw)
    add_normalized_metrics(fastmri)

    plot_metric_radar(
        m4raw,
        out_dir,
        "m4raw_efficiency",
        "M4Raw: normalized multi-metric profile",
        select_named(
            m4raw,
            [
                "AENet recurrent CNN",
                "VarNet non-shared CNN",
                "VarNet shared CNN",
                "AENet dual-domain",
                "AENet recurrent U-Net",
                "VarNet shared + CSME",
            ],
        ),
    )
    plot_rank_trajectory(
        m4raw,
        out_dir,
        "m4raw_rank",
        "M4Raw: metric-ranking trajectory",
    )
    plot_metric_radar(
        fastmri,
        out_dir,
        "fastmri_knee_efficiency",
        "fastMRI knee: normalized multi-metric profile",
        select_named(
            fastmri,
            [
                "AENet v2",
                "VarNet non-shared U-Net",
                "AENet recurrent U-Net",
                "FISTA-style",
                "VarNet shared CNN",
                "AENet dual-domain",
            ],
        ),
    )
    plot_rank_trajectory(
        fastmri,
        out_dir,
        "fastmri_knee_rank",
        "fastMRI knee: metric-ranking trajectory",
    )


if __name__ == "__main__":
    main()
