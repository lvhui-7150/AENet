"""Report the learned anchor and step-size schedules of trained AENet v2 runs.

Reads existing checkpoints only; no training is performed. The printed values
are the stage constants a_k and eta_k of the anchor--nullspace equivalence and
of the spectral damping corollary, i.e. the quantities reported in the paper's
table of learned anchor schedules.

Checkpoints are not distributed with this repository. Place them in
``checkpoints/`` (or pass ``--checkpoint-dir``) under their training names, e.g.
``m4raw_medium_recurrent_cnn_v2_s42_r4_s42_aenet_s42.pt``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_NAMES = {
    "M4Raw v2 (W48, D8, K12)": "m4raw_medium_recurrent_cnn_v2_s42_r4_s42_aenet_s42.pt",
    "fastMRI v2 seed 42 (W40, D4, K12)": "fastmri_knee_medium_aenet_v2_s42_r4_s42_aenet_s42.pt",
    "fastMRI v2 seed 123": "fastmri_knee_medium_aenet_v2_s123_r4_s42_aenet_s123.pt",
    "fastMRI v2 seed 2025": "fastmri_knee_medium_aenet_v2_s2025_r4_s42_aenet_s2025.pt",
    "BSD v2-large (W64, D3, K12)": "generic_inpainting_aenet_v2_s42.pt",
}


def softplus(value: torch.Tensor) -> torch.Tensor:
    return F.softplus(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints",
        help="directory holding the trained AENet v2 checkpoints",
    )
    args = parser.parse_args()

    anchors_seen: list[torch.Tensor] = []
    for name, filename in CHECKPOINT_NAMES.items():
        path = args.checkpoint_dir / filename
        if not path.exists():
            print(f"{name}: missing {path}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        anchors = state.get("log_anchor")
        steps = state.get("log_step")
        if anchors is None or steps is None:
            print(f"{name}: no log_anchor/log_step in state dict")
            continue
        a = softplus(anchors.float()).clamp(0.0, 1.0)
        eta = softplus(steps.float()).clamp(0.05, 2.0)
        lam = a / eta
        anchors_seen.append(a)
        fmt = lambda t: "[" + ", ".join(f"{v:.4f}" for v in t.tolist()) + "]"
        print(f"== {name}")
        print(f"   a_k        = {fmt(a)}")
        print(f"   eta_k      = {fmt(eta)}")
        print(f"   lambda_k   = {fmt(lam)}")
        print(f"   a range    = [{a.min():.4f}, {a.max():.4f}]  (mean {a.mean():.4f})")
        print(f"   lambda rng = [{lam.min():.4f}, {lam.max():.4f}]")

    if anchors_seen:
        stacked = torch.stack(anchors_seen)
        print("== summary over available runs")
        print(f"   a_k in [{stacked.min():.4f}, {stacked.max():.4f}]")


if __name__ == "__main__":
    main()
