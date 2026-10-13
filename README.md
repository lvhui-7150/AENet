# AENet: Anchored Nullspace-Aware Cascades for Rank-Deficient Inverse Problems

Code for the paper

> **Anchored Nullspace-Aware Cascades for Rank-Deficient Inverse Problems:
> Stage-Varying Stability Theory and Its Finite Realization**

The paper studies a finite, stage-varying recurrent cascade (**AENet v2**) for
rank-deficient linear inverse problems `y = Ax + eps` with possibly non-injective
`A`, whose data-visible/nullspace structure carries the stability certificate,
and whose tied stage map has an implicit-depth equilibrium envelope.

This repository contains the training, data-preparation, analysis, and figure
code for the three reported protocols: M4Raw multi-coil brain, fastMRI
single-coil knee, and BSD400/BSD68 natural-image inpainting.

## Repository layout

```
experiments/   training and data preparation
analysis/      result aggregation and the learned-anchor schedule
figures/       scripts that generate the paper's figures
configs/       the configuration of every reported run (JSON)
results/       per-run result JSONs behind the reported tables
hpc/           cluster job scripts and the fastMRI job generator
docs/          protocol notes for the M4Raw, fastMRI, and BSD runs
```

## Environment

Python 3.9 with PyTorch 2.4.1+cu121 was used for all reported runs.

```bash
pip install -r requirements.txt
```

## Data preparation

None of the datasets are redistributed here; download them from their original
releases.

* **M4Raw** (multi-coil brain): `python experiments/prepare_m4raw.py` reads the
  released volumes, estimates root-sum-of-squares normalized sensitivity maps
  from the central calibration region, and writes `m4raw_medium.pt`.
* **fastMRI knee** (single-coil): `python experiments/prepare_fastmri_knee.py`,
  or `python experiments/convert_fastmri_processed.py` for an already processed
  tensor, writes `fastmri_knee_singlecoil_medium.pt`.
* **BSD400/BSD68** (inpainting): use the BSD400 and BSD68 images from the
  "denoising datasets" release directly; the trainer reads the directory tree.

## Training the reported models

The commands below reproduce the reported AENet v2 runs. The baselines were
trained with the same data splits, masks, and evaluation protocol but with their
own capacity, batch size, and optimizer settings; each baseline configuration is
recorded in its result file under `results/`.

**M4Raw multi-coil brain** (recurrent CNN, width 48, backbone depth 8, 12 stages):

```bash
python experiments/train_aenet_m4raw.py \
  --data data/m4raw/prepared/m4raw_medium.pt \
  --arch recurrent_cnn --steps 12 --hidden 48 --depth 8 --anchor-init 0.05 \
  --coils 4 --augment --epochs 200 --batch-size 2 --grad-accum 4 \
  --lr 2e-4 --deep-supervision 0.15 --nmse-weight 1.0 \
  --ema-decay 0.99 --ema-start-epoch 40 \
  --seed 42 --only aenet --tag medium_recurrent_cnn_v2_s42 \
  --output results/m4raw_results_hpc_medium_recurrent_cnn_v2_s42.json
```

**fastMRI single-coil knee** (recurrent U-Net, width 40, backbone depth 4, 12 stages):

```bash
python experiments/train_aenet_m4raw.py \
  --data data/fastmri/prepared/fastmri_knee_singlecoil_medium.pt \
  --arch recurrent --steps 12 --hidden 40 --depth 4 --anchor-init 0.05 \
  --coils 1 --augment --epochs 200 --batch-size 4 --grad-accum 2 \
  --lr 2e-4 --deep-supervision 0.15 --nmse-weight 1.0 \
  --ema-decay 0.99 --ema-start-epoch 40 \
  --seed 42 --only aenet --tag fastmri_knee_medium_aenet_v2_s42 \
  --output results/fastmri_knee_results_hpc_medium_aenet_v2_s42.json
```

Use `--seed 123` and `--seed 2025` for the other two reported fastMRI seeds.

**BSD400/BSD68 inpainting** (recurrent CNN, width 48 or 64, backbone depth 3, 12 stages):

```bash
python experiments/train_generic_inpainting.py \
  --data-root data/inpainting/raw/denoising-datasets-main \
  --models aenet_v2 --steps 12 --hidden 48 --depth 3 \
  --epochs 150 --batch-size 4 --lr 2e-4 --mse-weight 0.1 \
  --deep-supervision 0.15 --augment --seed 42 \
  --output results/generic_inpainting_aenet_v2_results.json
```

Set `--hidden 64 --batch-size 2` for the reported high-capacity `v2-large` run.

## Reproducing the reported numbers

| Result | Source |
| --- | --- |
| Cross-dataset summary, per-dataset model zoos | `results/*.json` via `analysis/analyze_all_results_v1.py`, `analysis/analyze_m4raw_results_v7.py`, `analysis/analyze_fastmri_knee_results_v2.py` |
| Learned anchor schedule `a_k`, step size `eta_k`, `lambda_k = a_k/eta_k` | `analysis/report_learned_anchors.py --checkpoint-dir <dir>` |
| Paired / challenge comparisons against the strongest baselines | `analysis/analyze_aenet_v2_paired.py`, `analysis/analyze_aenet_v2_challenge.py` |
| Figures | `figures/make_paper_result_figures.py`, `figures/make_fastmri_reconstruction_figures.py`, `figures/make_fastmri_multimodel_comparison.py`, `figures/make_generic_inpainting_comparison.py` |

`configs/` holds the configuration of each reported run in isolation, e.g.
`configs/aenet_v2_m4raw_s42.json` and `configs/aenet_v2_large_bsd.json`.

### Reported results

| Dataset | AENet v2 | Strongest listed baseline | PSNR gain |
| --- | --- | --- | --- |
| M4Raw multi-coil | 30.956 dB | 30.796 dB | +0.159 dB |
| fastMRI single-coil knee | 35.491 dB | 35.043 dB | +0.448 dB |
| BSD400/BSD68 inpainting | 26.640 dB | 26.143 dB | +0.497 dB |

### Model summary

| Dataset | Instantiation | Width / backbone depth / stages | Parameters |
| --- | --- | --- | --- |
| M4Raw | recurrent CNN | 48 / 8 / 12 | 466,250 |
| fastMRI knee | recurrent U-Net | 40 / 4 / 12 | 16,522,354 |
| BSD400/BSD68 | recurrent CNN | 48 / 3 / 12 | 6,295,059 |
| BSD400/BSD68 (v2-large) | recurrent CNN | 64 / 3 / 12 | 11,184,803 |

## Notes

* **Checkpoints** are not stored in this repository. `analysis/report_learned_anchors.py`
  needs them; point `--checkpoint-dir` at a directory holding the run checkpoints
  under their training names (see the script docstring).
* **Third-party baselines.** The official fastMRI VarNet and HUMUS-Net baselines
  are evaluated through `experiments/external_adapter.py`, which wraps their
  public implementations. Those repositories are not vendored here; place them
  under `external/` as described in `docs/README_FASTMRI_KNEE.md`.
* **Scope.** The released training and evaluation path is the finite cascade. The
  implicit-depth equilibrium solver and implicit differentiation are analyzed in
  the paper and are not part of this code path.
* **Cluster scripts.** `hpc/` documents the scheduler jobs used for the reported
  runs. They contain `/path/to/aenet` placeholders for the project root and the
  conda environment; adjust them for your own setup.

## Citation

```bibtex
@article{aenet,
  title   = {Anchored Nullspace-Aware Cascades for Rank-Deficient Inverse Problems:
             Stage-Varying Stability Theory and Its Finite Realization},
  author  = {Anonymous},
  journal = {under review},
  year    = {2026}
}
```

## License

Not yet specified. Please contact the authors before reusing the code.
