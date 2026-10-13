# M4Raw Baseline Suite

## Model-Zoo Principle

AENet is one entry in a model zoo. VarNet is one baseline in that zoo, not
the only comparison target. Every model must be evaluated under the same
split, mask, preprocessing, training budget, and evaluation code.

The main comparison table should contain:

- classical reconstruction controls,
- unrolled VarNet-family controls,
- official external baselines,
- implicit-depth and monotone-operator controls,
- AENet variants.

Published fastMRI numbers must be kept in a separate contextual table.

## Protocol Rule

All models in the main comparison must use the same split, mask, resolution,
coil-sensitivity preprocessing, and evaluation code. Published fastMRI numbers
belong in a separate contextual table.

## Tier 0: Implemented in This Repository

| ID | Model | Script | Status |
|---|---|---|---|
| B0 | Zero-filled | built into `train_aenet_m4raw.py` | done |
| B1 | Shared CNN VarNet | `run_aenet_m4raw_medium_varnet_shared.sh` | done |
| B2 | Non-shared CNN VarNet | `run_aenet_m4raw_medium_varnet.sh` | done |
| B3 | Recurrent CNN AENet | `run_aenet_m4raw_medium_recurrent_cnn_aenet.sh` | done |
| B4 | Adaptive recurrent CNN AENet | `run_aenet_m4raw_medium_adaptive_recurrent_cnn_aenet.sh` | done |
| B5 | U-Net shared VarNet | `run_aenet_m4raw_medium_unet_varnet_shared.sh` | done |
| B6 | U-Net non-shared VarNet | `run_aenet_m4raw_medium_unet_varnet.sh` | done |
| B7 | Dual-domain shared VarNet | `run_aenet_m4raw_medium_dual_domain_shared.sh` | done |
| B8 | Dual-domain recurrent AENet | `run_aenet_m4raw_medium_dual_domain_aenet.sh` | done |
| B9 | Shared VarNet + smap refinement | `run_m4raw_medium_smaprefine_shared.sh` | running |
| B10 | Feature-image VarNet | `run_m4raw_medium_feature_image_shared.sh` | done |
| B11 | Feature-image + smap refinement | `run_m4raw_medium_feature_image_smaprefine_shared.sh` | running |

Current reference results from the medium pilot:

| Method | Test PSNR | Test SSIM | Test NMSE |
|---|---:|---:|---:|
| VarNet non-shared CNN | 30.796 | 0.7631 | 0.0234 |
| VarNet shared CNN | 30.705 | 0.7616 | 0.0238 |
| AENet recurrent CNN | 30.700 | 0.7610 | 0.0239 |
| Feature-image shared | 30.552 | 0.7575 | 0.0247 |
| Adaptive recurrent CNN AENet | 30.546 | 0.7571 | 0.0247 |

## Tier 1: Required Strong Baselines

| Priority | Model | Source | What to port | Role |
|---|---|---|---|---|
| 1 | E2E VarNet | fastMRI official code | cascade U-Net, learned sensitivity estimation, Cartesian DC, same mask | mandatory |
| 2 | HUMUS-Net | public code | multi-scale CNN, low-resolution Transformer, unrolled DC | strong |
| 3 | FI VarNet | Scientific Reports 2024 | feature-domain propagation, artifact-aware attention, feature-image fusion | strong |
| 4 | SDUM-style model | arXiv 2512.17137 | learned CSME, sampling-aware weighted DC, cascade conditioning | strong |
| 5 | SPICER-style model | arXiv 2210.02584 | automatic coil-sensitivity estimation, self-supervised objective | ablation |
| 6 | MOL | paper and reference code | monotone operator learning with learned regularization | theory control |
| 7 | DEQ baseline | paper and reference code | implicit reconstruction with fixed-point solve | depth control |

## Tier 2: Multi-Contrast Baselines

| Priority | Model | Source | Requirement | Role |
|---|---|---|---|---|
| 1 | DSFormer | arXiv 2201.10776 | aligned T1/T2/FLAIR volumes from the same subject | multi-contrast |
| 2 | CoSMo-RecNet | arXiv 2609.01959 | unpaired multi-contrast prior plus aligned reference contrast | low-data |

The current medium `.pt` file does not preserve subject/contrast pairing, so a
new preparation script is required before these models can be trained.

## Tier 3: High-Cost Baselines

| Model | Source | Main cost |
|---|---|---|
| W-UNet Variational Network | arXiv 2606.15167 | wavelet U-Net and sensitivity estimation |
| HiFi-MambaV2 | arXiv 2511.18534 | Mamba/MoE implementation and large compute |
| AIRS-Net | leaderboard entry | public paper/code currently not located |
| DIRCN | DOI 10.1007/s10334-022-01041-3 | separate dual-domain implementation |

## Recommended Execution Order

1. Finish B9 and B11.
2. Port and run official E2E VarNet.
3. Port and run HUMUS-Net.
4. Port FI VarNet or promote a faithful FI implementation.
5. Add learned CSME from SDUM or SPICER to the best Tier 1 baseline.
6. Add multi-contrast DSFormer or CoSMo-RecNet after preparing aligned data.
7. Add W-UNet and HiFi-MambaV2 only if compute budget remains.

## AENet Integration Rule

AENet is trained only after the strongest same-protocol baseline is fixed.
The first AENet integration must be a warm-start:

1. load the best strong-baseline checkpoint,
2. freeze the backbone,
3. train only AENet anchor, step, and nullspace correction parameters,
4. keep the baseline weights as a fallback,
5. accept AENet only if validation and test improve consistently.

## Decision Gates

```text
Gate 1:
  Strong baseline must beat shared CNN VarNet (30.705 dB).

Gate 2:
  AENet warm-start must beat the strong baseline by >= 0.10 dB
  with SSIM and NMSE moving in the same direction.

Gate 3:
  Re-run the winning pair with at least 3 seeds and 3 masks.

Gate 4:
  Only then expand from medium to the full official M4Raw split.
```
