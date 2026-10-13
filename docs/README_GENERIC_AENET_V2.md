# Generic-image AENet v2 patch

This patch adds a stronger generic-image AENet for BSD400/BSD68 inpainting.

Main changes:

```text
- Wider recurrent prior: hidden 48, depth 3
- 12 equilibrium steps
- Learnable anchor at every step
- Per-channel step scale and bias
- Deep supervision over intermediate estimates
```

## Install

```bash
cp -a AENet_generic_patch_20261009_v3/experiments/. \
  /path/to/aenet/AENet/experiments/

cp -a AENet_generic_patch_20261009_v3/hpc/. \
  /path/to/aenet/AENet/hpc/
```

## Run

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_generic_inpainting_aenet_v2.sh
jjobs
```

Expected result:

```text
/path/to/aenet/AENet/experiments/generic_inpainting_aenet_v2_results.json
```

Target:

```text
Beat the current non-shared unrolled model:
PSNR > 26.143 dB
```

If the primary model does not reach that target, run the high-capacity
fallback with hidden 64:

```bash
jsub < run_generic_inpainting_aenet_v2_large.sh
```
