"""Compare AENet v2 seeds with the current fastMRI knee leader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_model(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_name = payload["config"]["only"]
    model = payload["models"][model_name]
    return payload, model


def metric(model, key):
    return float(model["test"][key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument(
        "--leader",
        default="fastmri_knee_results_hpc_medium_unet_varnet.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    leader_payload, leader = load_model(result_dir / args.leader)
    leader_metrics = {
        "psnr": metric(leader, "psnr"),
        "ssim": metric(leader, "ssim"),
        "nmse": metric(leader, "nmse"),
    }

    v2_paths = sorted(result_dir.glob("fastmri_knee_results_hpc_medium_aenet_v2_s*.json"))
    if not v2_paths:
        raise RuntimeError("no AENet v2 result JSON files were found")

    v2_runs = []
    for path in v2_paths:
        _, model = load_model(path)
        v2_runs.append(
            {
                "name": path.stem.split("medium_")[-1],
                "params": model["params"],
                "best_epoch": model["best_val"]["epoch"],
                "psnr": metric(model, "psnr"),
                "ssim": metric(model, "ssim"),
                "nmse": metric(model, "nmse"),
                "train_hours": model["train_sec"] / 3600.0,
            }
        )

    means = {
        key: float(np.mean([run[key] for run in v2_runs]))
        for key in ("psnr", "ssim", "nmse")
    }
    stds = {
        key: float(np.std([run[key] for run in v2_runs], ddof=1))
        if len(v2_runs) > 1
        else 0.0
        for key in ("psnr", "ssim", "nmse")
    }

    psnr_gain = means["psnr"] - leader_metrics["psnr"]
    ssim_gain = means["ssim"] - leader_metrics["ssim"]
    nmse_gain = leader_metrics["nmse"] - means["nmse"]

    lines = [
        "# AENet v2 fastMRI Knee Challenge",
        "",
        "## Current Leader",
        "",
        f"- `{args.leader}`",
        f"- PSNR: {leader_metrics['psnr']:.6f}",
        f"- SSIM: {leader_metrics['ssim']:.6f}",
        f"- NMSE: {leader_metrics['nmse']:.6f}",
        "",
        "## AENet v2 Seeds",
        "",
        "| Run | Params | Best Epoch | PSNR | SSIM | NMSE | Hours |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in v2_runs:
        lines.append(
            f"| {run['name']} | {run['params']:,} | {run['best_epoch']} | "
            f"{run['psnr']:.6f} | {run['ssim']:.6f} | {run['nmse']:.6f} | "
            f"{run['train_hours']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Mean Result",
            "",
            f"- PSNR: {means['psnr']:.6f} +/- {stds['psnr']:.6f}",
            f"- SSIM: {means['ssim']:.6f} +/- {stds['ssim']:.6f}",
            f"- NMSE: {means['nmse']:.6f} +/- {stds['nmse']:.6f}",
            "",
            "## Comparison",
            "",
            f"- PSNR gain over leader: {psnr_gain:+.6f} dB",
            f"- SSIM gain over leader: {ssim_gain:+.6f}",
            f"- NMSE reduction over leader: {nmse_gain:+.6f}",
            "",
            (
                "VERDICT: AENet v2 beats the leader on all three metrics."
                if psnr_gain > 0 and ssim_gain > 0 and nmse_gain > 0
                else "VERDICT: AENet v2 does not yet beat the leader on all metrics."
            ),
        ]
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
