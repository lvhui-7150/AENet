"""Compute paired slice-level AENet v2 versus VarNet comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from make_fastmri_reconstruction_figures import (
    ComplexUNet,
    RecurrentAENetMC,
    VarNetMC,
    make_inputs,
)
from train_tpami import metric_batch


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"], payload["config"], payload["mask"]


def build_aenet(checkpoint_path: Path):
    state, config, mask = load_checkpoint(checkpoint_path)
    model = RecurrentAENetMC(
        mask,
        steps=int(config["steps"]),
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        anchor_init=float(config["anchor_init"]),
    )
    model.load_state_dict(state)
    return model, mask.float()


def build_varnet(checkpoint_path: Path):
    state, config, mask = load_checkpoint(checkpoint_path)
    model = VarNetMC(
        mask,
        steps=int(config["steps"]),
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        prior_factory=ComplexUNet,
    )
    model.load_state_dict(state)
    return model, mask.float()


@torch.no_grad()
def evaluate(model, data_path: Path, mask: torch.Tensor, device: torch.device):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    model.eval().to(device)
    mask = mask.to(device)
    psnr, ssim, nmse = [], [], []
    for index in data["split"]["test"]:
        gt_batch = data["gt_complex"][index : index + 1]
        smap_batch = data["smap"][index : index + 1]
        kspace_batch = data["kspace"][index : index + 1]
        gt, y, z0 = make_inputs(gt_batch, smap_batch, kspace_batch, mask, device)
        recon = model(z0, y, smap_batch.to(device))
        metrics = metric_batch(recon, gt)
        psnr.append(float(metrics["psnr"][0]))
        ssim.append(float(metrics["ssim"][0]))
        nmse.append(float(metrics["nmse"][0]))
    return {
        "psnr": np.asarray(psnr),
        "ssim": np.asarray(ssim),
        "nmse": np.asarray(nmse),
    }


def bootstrap_mean(values, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compare(reference, candidate, samples: int, seed: int):
    output = {}
    for metric in ("psnr", "ssim", "nmse"):
        if metric == "nmse":
            difference = reference[metric] - candidate[metric]
        else:
            difference = candidate[metric] - reference[metric]
        t_stat, t_p = stats.ttest_rel(candidate[metric], reference[metric])
        try:
            w_stat, w_p = stats.wilcoxon(
                candidate[metric],
                reference[metric],
                alternative="two-sided",
            )
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        low, high = bootstrap_mean(difference, samples, seed)
        output[metric] = {
            "mean_difference": float(difference.mean()),
            "median_difference": float(np.median(difference)),
            "std_difference": float(difference.std(ddof=1)),
            "bootstrap_95ci": [float(low), float(high)],
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "wins": int(np.sum(difference > 0)),
            "ties": int(np.sum(difference == 0)),
            "losses": int(np.sum(difference < 0)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--leader-checkpoint", required=True)
    parser.add_argument("--aenet-checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    leader, leader_mask = build_varnet(Path(args.leader_checkpoint))
    leader_metrics = evaluate(leader, Path(args.data), leader_mask, device)

    report = {"leader": args.leader_checkpoint, "seeds": {}}
    for index, checkpoint in enumerate(args.aenet_checkpoints):
        candidate, candidate_mask = build_aenet(Path(checkpoint))
        if not torch.equal(leader_mask, candidate_mask):
            raise RuntimeError(f"mask mismatch for {checkpoint}")
        candidate_metrics = evaluate(candidate, Path(args.data), candidate_mask, device)
        report["seeds"][Path(checkpoint).stem] = {
            "mean_metrics": {
                key: float(value.mean()) for key, value in candidate_metrics.items()
            },
            "comparison": compare(
                leader_metrics,
                candidate_metrics,
                args.bootstrap_samples,
                args.seed + index,
            ),
        }
        print(f"[done] {Path(checkpoint).name}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
