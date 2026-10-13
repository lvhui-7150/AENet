# fastMRI Knee Model-Zoo Runbook

This package runs the current M4Raw model zoo on fastMRI knee data. The
single-coil pipeline is the default because the earlier fastMRI development
results used `knee_singlecoil` at 128 x 128.

## 1. Data Layout

The fastMRI data agreement is required before downloading the official files.
After extraction, the HPC layout should be:

```text
/path/to/aenet/data/fastmri/
  knee_singlecoil_train/    # .h5 files, nested directories are allowed
  knee_singlecoil_val/      # .h5 files, nested directories are allowed
  prepared/
```

If the previously prepared tensors are used, the smaller legacy layout is:

```text
/path/to/aenet/data/fastmri/
  legacy/
    fastmri_128_prepared.pt
    fastmri_320/
      fastmri_320_meta.pt
      fastmri_320_gt_chunk_000.pt
      ...
      fastmri_320_gt_chunk_007.pt
  prepared/
```

Typical archive names are:

```text
knee_singlecoil_train.tar.xz
knee_singlecoil_val.tar.xz
```

Example extraction:

```bash
cd /path/to/aenet/data/fastmri
tar -xf knee_singlecoil_train.tar.xz
tar -xf knee_singlecoil_val.tar.xz
```

The preparation script searches recursively, so either
`knee_singlecoil_val/*.h5` or `knee_singlecoil_val/singlecoil_val/*.h5`
works.

## 2. Minimal Upload List

If the existing AENet code is already on the HPC, upload or replace:

```text
experiments/train_aenet_m4raw.py
experiments/prepare_fastmri_knee.py
experiments/convert_fastmri_processed.py
hpc/run_prepare_fastmri_knee_medium.sh
hpc/run_prepare_fastmri_knee_native320.sh
hpc/run_prepare_fastmri_knee_multicoil_medium.sh
hpc/run_convert_fastmri_128_processed.sh
hpc/run_convert_fastmri_320_processed.sh
hpc/generated_fastmri_knee_medium/
external_code.tar.gz
```

The generated directory contains the 22 model scripts and one submission
script. Do not upload `__pycache__` or local checkpoints.

For the first experiment, the only data file to upload is the 26 MB
`fastmri_128_prepared.pt` described in Section 3A.

## 3A. Use the Existing 128 Prepared Tensor

This is the recommended first experiment. Upload only:

```text
fastmri_128_prepared.pt
```

to:

```text
/path/to/aenet/data/fastmri/legacy/fastmri_128_prepared.pt
```

Convert it to the current model-zoo format:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_convert_fastmri_128_processed.sh
```

The converted file is:

```text
/path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_medium.pt
```

It contains 199 slices with the existing 159/20/20 train/val/test split and
six masks. The conversion reconstructs `kspace` and `smap` from the complex
ground-truth images, so uploading the 19 GB raw H5 files is not required for
the first 128 x 128 experiment. These 199 slices come from the official
validation volumes. No official single-coil knee training H5 directory was
found on the local C, D, E, or F drives.

## 3B. Use the Existing 320 Chunked Tensor

Upload `fastmri_320_meta.pt` and all eight
`fastmri_320_gt_chunk_000.pt` to `fastmri_320_gt_chunk_007.pt` files to:

```text
/path/to/aenet/data/fastmri/legacy/fastmri_320/
```

Convert a 1200-slice pilot:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_convert_fastmri_320_processed.sh
```

Output:

```text
/path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_native320.pt
```

The corrupt archived file `_archive/fastmri_320_prepared.pt` must not be
uploaded. Use only the meta file and the eight verified chunks.

## 3C. Alternative: Prepare from Raw H5

This path is only needed when the prepared tensors are unavailable.

Submit from the login node:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_prepare_fastmri_knee_medium.sh
```

Expected output:

```text
/path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_medium.pt
/path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_medium.json
```

The raw-H5 pilot contains 1200 slices:

```text
train: 200 volumes x 4 slices = 800
val:    50 volumes x 4 slices = 200
test:   50 volumes x 4 slices = 200
```

The val/test split is made at the volume level from the official validation
directory. The public fastMRI test directory does not contain ground-truth
images, so it is not used for model selection.

Check the preparation result:

```bash
tail -f /path/to/aenet/AENet/fastmri_knee_prepare_medium_*.out

/path/to/aenet/miniconda3/envs/aenet/bin/python -c \
"import torch; p='/path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_medium.pt'; \
d=torch.load(p,map_location='cpu',weights_only=False); \
print(d['gt_complex'].shape, d['smap'].shape, d['kspace'].shape); \
print({k:len(v) for k,v in d['split'].items()})"
```

The expected shape pattern is:

```text
gt_complex: (1200, 128, 128)
smap:       (1200, 1, 128, 128)
kspace:     (1200, 1, 128, 128)
```

## 4. Run the Model Zoo

Submit all 22 jobs:

```bash
cd /path/to/aenet/AENet/hpc/generated_fastmri_knee_medium
bash submit_all_fastmri_knee_medium.sh
```

Or submit selected models:

```bash
jsub < run_fastmri_knee_medium_recurrent_cnn_aenet.sh
jsub < run_fastmri_knee_medium_varnet.sh
jsub < run_fastmri_knee_medium_official_varnet.sh
```

Check the queue:

```bash
jjobs
```

Watch one job:

```bash
tail -f /path/to/aenet/AENet/fastmri_knee_recurrent_cnn_aenet_<JOBID>.out
```

All jobs use `ai_share`, one GPU, 100 epochs, seed 42, and the `r4_s42`
mask. Hyperparameters match the corresponding M4Raw medium scripts wherever
the architecture is the same.

## 5. AENet v2 Challenge Run

The first fastMRI run placed AENet second, 0.0335 dB behind the non-shared
VarNet U-Net. The AENet v2 recipe targets that PSNR/NMSE gap while preserving
the SSIM advantage:

```text
recurrent U-Net: hidden 40, depth 4, steps 12
training: 200 epochs, 3 seeds, EMA, extra NMSE loss term
```

Submit the three seeds in one batch:

```bash
cd /path/to/aenet/AENet/hpc
BATCH_SIZE=5 bash run_aenet_v2_seeds_batched.sh
```

Use `BATCH_SIZE=2` if the shared GPU is occupied.

After all three result JSON files are copied to the result directory, run:

```bash
/path/to/aenet/miniconda3/envs/aenet/bin/python \
  experiments/analyze_aenet_v2_challenge.py \
  --result-dir experiments \
  --out experiments/aenet-v2-challenge.md
```

The report compares the three-seed AENet v2 mean with
`unet_varnet` on PSNR, SSIM, and NMSE.

If the three-seed v2 mean remains slightly below the leader, run the larger
fallback:

```bash
jsub < run_fastmri_knee_medium_aenet_v3_s42.sh
```

AENet v3 uses hidden width 48, depth 4, and 16 equilibrium steps. Its
parameter count is approximately 23.8M, still below the 26.2M-parameter
non-shared VarNet U-Net leader.

## 6. Results

Each job writes:

```text
experiments/fastmri_knee_results_hpc_medium_<model>.json
```

Check completed JSON files:

```bash
cd /path/to/aenet/AENet
ls -lh experiments/fastmri_knee_results_hpc_medium_*.json
```

If some jobs exited abnormally, run the diagnostic summary:

```bash
cd /path/to/aenet/AENet/hpc
bash diagnose_fastmri_knee_jobs.sh
```

The script prints the current queue, a table of OK/FAIL/UNKNOWN jobs, the last
120 error lines for every failed job, and matches common causes such as CUDA
out-of-memory, missing modules, missing files, and invalid arguments.

When the failure message is `CUDA out of memory` and another process is using
most of the GPU, audit the worker first:

```bash
cd /path/to/aenet/AENet/hpc
bash check_hpc_gpu_availability.sh
jsub < run_fastmri_gpu_audit.sh
tail -f /path/to/aenet/AENet/fastmri_gpu_audit_*.out
```

`check_hpc_gpu_availability.sh` prints `jjobs` and probes this scheduler's
supported host/GPU query commands. `run_fastmri_gpu_audit.sh` runs on an
allocated worker and prints every GPU on that host, including UUID, total/used
memory, PIDs, users, elapsed time, and commands.

The audit reports the GPU UUID, total/used/free memory, compute-process PIDs,
users, elapsed time, and command lines. If the dominant process is a stale job
owned by the same account, stop that job before resubmitting. If it belongs to
another user, wait or use another GPU.

### CPU Fallback

When no GPU is usable, AENet can run on `cnode48` without requesting a GPU.
The original recurrent U-Net is the cheaper CPU fallback:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_fastmri_knee_medium_recurrent_aenet_cpu.sh
```

The larger AENet v2 CPU run is also available:

```bash
jsub < run_fastmri_knee_medium_aenet_v2_s42_cpu.sh
```

CPU runs use 16 threads and no GPU. They are much slower, but they do not
fail because of GPU contention. The official E2E VarNet and HUMUS-Net
adapters should remain GPU jobs because their CPU training is substantially
slower.

To rerun only the jobs that failed because of shared-GPU contention, submit
them in batches of five:

```bash
cd /path/to/aenet/AENet/hpc
BATCH_SIZE=5 bash run_failed_fastmri_jobs_batched.sh
```

Each batch waits for all five jobs to leave `jjobs` before the next batch is
submitted. To use a safer batch size, set `BATCH_SIZE=2` or `BATCH_SIZE=3`.

To run every generated fastMRI knee model in batches of five:

```bash
cd /path/to/aenet/AENet/hpc
BATCH_SIZE=5 bash run_all_fastmri_knee_jobs_batched.sh
```

Generate the ranking table and figures directly on the HPC:

```bash
cd /path/to/aenet/AENet
/path/to/aenet/miniconda3/envs/aenet/bin/python \
  experiments/analyze_fastmri_knee_results.py \
  --result-dir experiments \
  --out-dir analysis-fastmri-knee-medium
```

This writes `summary.csv`, `analysis-report.md`, and the three standard
figures. Collect the JSON files and this analysis directory locally afterward.

## 7. Native 320 x 320 and Multicoil

Native 320 preparation:

```bash
jsub < run_prepare_fastmri_knee_native320.sh
```

Then regenerate the jobs with the 320 data file:

```bash
cd /path/to/aenet/AENet/hpc
python make_fastmri_knee_jobs.py \
  --data-file /path/to/aenet/data/fastmri/prepared/fastmri_knee_singlecoil_native320.pt \
  --out-dir generated_fastmri_knee_native320
```

Multicoil preparation is optional:

```bash
jsub < run_prepare_fastmri_knee_multicoil_medium.sh
```

For multicoil, replace `--coils 1` with the coil count printed in the
preparation JSON before generating jobs. The prepared tensor pads shorter
volumes with zero-filled coil channels.

## 8. Important Evaluation Notes

- The 128 x 128 pipeline center-crops the image after the inverse FFT and
  then constructs an image-domain k-space. This matches the earlier fastMRI
  128 development convention.
- The R=4 mask uses exactly `round(width / 4)` sampled phase-encode lines,
  with an 8% fully sampled center band and symmetric random outer lines.
- `Official E2E VarNet` is designed for multi-coil data. Its single-coil
  result must be inspected and marked invalid if it falls below the
  zero-filled control, as happened on M4Raw.
- `Official HUMUS-Net` uses checkpointing and may still be memory-heavy at
  320 x 320. Keep the 128 x 128 pilot as the first stable comparison.
