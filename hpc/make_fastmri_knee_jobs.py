"""Generate HPC job scripts for the fastMRI knee model zoo.

The generated scripts preserve the M4Raw model-zoo hyperparameters and only
change the data file, coil count, output paths, and log names.
"""

from __future__ import annotations

import argparse
from pathlib import Path


JOBS = [
    {
        "slug": "aenet",
        "arch": "cnn",
        "only": "aenet",
        "batch": 8,
        "accum": 1,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [],
    },
    {
        "slug": "varnet_shared",
        "arch": "cnn",
        "only": "varnet_shared",
        "batch": 8,
        "accum": 1,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [],
    },
    {
        "slug": "varnet",
        "arch": "cnn",
        "only": "varnet",
        "batch": 8,
        "accum": 1,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [],
    },
    {
        "slug": "recurrent_aenet",
        "arch": "recurrent",
        "only": "aenet",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--anchor-init 0.05", "--augment"],
    },
    {
        "slug": "unet_varnet_shared",
        "arch": "unet",
        "only": "varnet_shared",
        "batch": 8,
        "accum": 1,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "unet_varnet",
        "arch": "unet",
        "only": "varnet",
        "batch": 2,
        "accum": 4,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "recurrent_cnn_aenet",
        "arch": "recurrent_cnn",
        "only": "aenet",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": ["--anchor-init 0.05", "--augment"],
    },
    {
        "slug": "adaptive_cnn_aenet",
        "arch": "adaptive_recurrent_cnn",
        "only": "aenet",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": ["--anchor-init 0.05", "--augment"],
    },
    {
        "slug": "dual_domain_aenet",
        "arch": "dual_domain_recurrent_cnn",
        "only": "aenet",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": ["--anchor-init 0.02", "--augment"],
    },
    {
        "slug": "feature_image_shared",
        "arch": "feature_image_varnet",
        "only": "varnet_shared",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": ["--feature-channels 32", "--augment"],
    },
    {
        "slug": "fi_attention_shared",
        "arch": "feature_image_attention",
        "only": "varnet_shared",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": ["--feature-channels 32", "--augment"],
    },
    {
        "slug": "smaprefine_shared",
        "arch": "cnn",
        "only": "varnet_shared",
        "batch": 8,
        "accum": 1,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [
            "--smap-refine",
            "--smap-hidden 32",
            "--smap-depth 3",
            "--augment",
        ],
    },
    {
        "slug": "feature_image_smaprefine_shared",
        "arch": "feature_image_varnet",
        "only": "varnet_shared",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [
            "--feature-channels 32",
            "--smap-refine",
            "--smap-hidden 32",
            "--smap-depth 3",
            "--augment",
        ],
    },
    {
        "slug": "official_varnet",
        "arch": "official_varnet",
        "only": "varnet_shared",
        "batch": 4,
        "accum": 2,
        "steps": 12,
        "hidden": 32,
        "depth": 6,
        "lr": "2e-4",
        "extras": [
            "--official-varnet-cascades 12",
            "--official-varnet-chans 18",
            "--augment",
        ],
    },
    {
        "slug": "official_humusnet",
        "arch": "official_humusnet",
        "only": "varnet_shared",
        "batch": 1,
        "accum": 8,
        "steps": 8,
        "hidden": 32,
        "depth": 6,
        "lr": "1e-4",
        "extras": [
            "--official-humus-cascades 8",
            "--official-humus-embed 66",
            "--official-humus-checkpoint",
            "--augment",
        ],
    },
    {
        "slug": "direct_unet",
        "arch": "direct_unet",
        "only": "baseline",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "modl",
        "arch": "modl_style",
        "only": "baseline",
        "batch": 2,
        "accum": 4,
        "steps": 6,
        "hidden": 32,
        "depth": 3,
        "lr": "1e-4",
        "extras": [
            "--modl-steps 6",
            "--cg-steps 3",
            "--augment",
        ],
    },
    {
        "slug": "ista_soft",
        "arch": "ista_soft",
        "only": "baseline",
        "batch": 8,
        "accum": 1,
        "steps": 12,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "fista",
        "arch": "fista_style",
        "only": "baseline",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "admm",
        "arch": "admm_style",
        "only": "baseline",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "lpd",
        "arch": "lpd_net",
        "only": "baseline",
        "batch": 4,
        "accum": 2,
        "steps": 8,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
    {
        "slug": "attention_unet",
        "arch": "attention_unet",
        "only": "baseline",
        "batch": 4,
        "accum": 2,
        "steps": 10,
        "hidden": 32,
        "depth": 3,
        "lr": "2e-4",
        "extras": ["--augment"],
    },
]


def render_job(job: dict, project_root: str, data_file: str) -> str:
    extras = "\n".join(f"  {item} \\" for item in job["extras"])
    if extras:
        extras += "\n"
    return f"""#!/bin/bash
# fastMRI knee model-zoo job: {job['slug']}
# Submit with: jsub < run_fastmri_knee_medium_{job['slug']}.sh
#JSUB -q ai_share
#JSUB -gpgpu 1
#JSUB -n 4
#JSUB -J fmk_{job['slug']}
#JSUB -o {project_root}/fastmri_knee_{job['slug']}_%J.out
#JSUB -e {project_root}/fastmri_knee_{job['slug']}_%J.err

export PATH="/path/to/aenet/miniconda3/envs/aenet/bin:$PATH"
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/path/to/aenet/miniconda3/envs/aenet/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${{LD_LIBRARY_PATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
FASTMRI_ROOT="${{FASTMRI_ROOT:-/path/to/aenet/data/fastmri}}"
FASTMRI_DATA="${{FASTMRI_DATA:-{data_file}}}"
cd {project_root}

/path/to/aenet/miniconda3/envs/aenet/bin/python -u experiments/train_aenet_m4raw.py \\
  --data "${{FASTMRI_DATA}}" \\
  --mask-key r4_s42 \\
  --epochs 100 \\
  --batch-size {job['batch']} \\
  --grad-accum {job['accum']} \\
  --steps {job['steps']} \\
  --hidden {job['hidden']} \\
  --depth {job['depth']} \\
  --arch {job['arch']} \\
  --coils 1 \\
  --lr {job['lr']} \\
{extras}  --only {job['only']} \\
  --tag fastmri_knee_medium_{job['slug']} \\
  --output experiments/fastmri_knee_results_hpc_medium_{job['slug']}.json
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default="/path/to/aenet/AENet",
    )
    parser.add_argument(
        "--data-file",
        default="/path/to/aenet/data/fastmri/prepared/"
        "fastmri_knee_singlecoil_medium.pt",
    )
    parser.add_argument(
        "--out-dir",
        default="generated_fastmri_knee_medium",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for job in JOBS:
        path = out_dir / f"run_fastmri_knee_medium_{job['slug']}.sh"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_job(job, args.project_root, args.data_file))
        generated.append(path.name)

    submit_path = out_dir / "submit_all_fastmri_knee_medium.sh"
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        "",
    ]
    for name in generated:
        submit_lines.append(f"jsub < {name}")
        submit_lines.append("sleep 2")
    with submit_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(submit_lines) + "\n")
    print(f"[done] generated {len(generated)} jobs in {out_dir}")
    print(f"[done] generated {submit_path}")


if __name__ == "__main__":
    main()
