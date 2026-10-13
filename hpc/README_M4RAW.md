# M4Raw Brain preparation and AENet run on the CDUT HPC

This downloads the open M4Raw Brain multicoil H5 archives from Zenodo,
builds a prepared smoke dataset, and runs the multicoil AENet pipeline.
No registration or fastMRI data agreement is required.

## 1. Files to upload

Upload the scripts below to `/path/to/aenet/AENet/` using the portal
file manager, SFTP, or Xftp. Keep the directory layout intact.

```text
AENet/
  experiments/
    prepare_m4raw.py
    train_aenet_m4raw.py
    train_tpami.py
  hpc/
    download_m4raw.sh
    run_download_m4raw.sh
    run_prepare_m4raw_smoke.sh
    run_aenet_m4raw_smoke.sh
```

Line endings must stay LF. If the file manager changes the `.sh` files to
CRLF, run this once on the cluster:

```bash
cd /path/to/aenet/AENet/hpc
sed -i 's/\r$//' *.sh
```

## 2. What gets downloaded

By default the job downloads and extracts train + val + test:

```text
/path/to/aenet/data/m4raw/
  zips/
    M4RawV1.5_multicoil_train.zip     12.7 GB
    M4RawV1.5_multicoil_val.zip        3.0 GB
    M4Raw_multicoil_test.zip           5.0 GB
  raw/
    M4RawV1.5_multicoil_train/
      multicoil_train/*.h5
    M4RawV1.5_multicoil_val/
      multicoil_val/*.h5
    M4Raw_multicoil_test/
      test/*.h5
```

The default total is about 20.7 GB of zip data. The optional GRE archive is
another 4.8 GB.

## 3. Directory layout

```text
/path/to/aenet/AENet/
  experiments/
    prepare_m4raw.py
    train_aenet_m4raw.py
    train_tpami.py
  hpc/
    download_m4raw.sh
    run_download_m4raw.sh
    run_prepare_m4raw_smoke.sh
    run_aenet_m4raw_smoke.sh

/path/to/aenet/data/m4raw/
  zips/
    M4RawV1.5_multicoil_train.zip
    M4RawV1.5_multicoil_val.zip
    M4Raw_multicoil_test.zip
  raw/
    multicoil_train/*.h5
    multicoil_val/*.h5
    multicoil_test/*.h5
  prepared/
    m4raw_smoke.pt
```

Create the directories:

```bash
mkdir -p /path/to/aenet/AENet/experiments
mkdir -p /path/to/aenet/AENet/hpc
mkdir -p /path/to/aenet/data/m4raw/{zips,raw,prepared}
```

## 4. Submit

First download train/val/test from Zenodo on a CPU node:

From a cluster Terminal:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_download_m4raw.sh
jjobs
```

Output and errors:

```bash
cat /path/to/aenet/AENet/m4raw_download_*.out
cat /path/to/aenet/AENet/m4raw_download_*.err
```

After the download finishes, prepare the 96-slice smoke file:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_prepare_m4raw_smoke.sh
cat /path/to/aenet/AENet/m4raw_prepare_smoke_*.out
cat /path/to/aenet/AENet/m4raw_prepare_smoke_*.err
```

Then start the GPU smoke run:

```bash
cd /path/to/aenet/AENet/hpc
jsub < run_aenet_m4raw_smoke.sh
cat /path/to/aenet/AENet/aenet_m4raw_smoke_*.out
cat /path/to/aenet/AENet/aenet_m4raw_smoke_*.err
```

To watch a live job in another Terminal:

```bash
tail -f /path/to/aenet/AENet/m4raw_download_*.out
```

If the download is interrupted, submit the same job again. `curl -C -`
resumes partial files, and every archive is verified with `md5sum` before
extraction.

## 5. Check result

The smoke result is written to:

```text
/path/to/aenet/AENet/experiments/m4raw_results_hpc_smoke.json
```

The checkpoints are written to:

```text
/path/to/aenet/AENet/experiments/m4raw_r4_s42_*_s42.pt
```

## 6. Queue notes

`run_download_m4raw.sh` uses `#JSUB -q cnode48`. If that CPU queue is not
available, check your queues and switch to another CPU queue:

```bash
jqueues -u "$USER" -l
```

Then edit only the `#JSUB -q` line, for example `cnode56` or `fat`.

## 7. Optional: include GRE

Edit `run_download_m4raw.sh` and change the final line to:

```bash
bash download_m4raw.sh train val test gre
```

You can also run the download script with only selected parts:

```bash
bash download_m4raw.sh train val
```

## 8. Offline option

If compute nodes cannot reach Zenodo, upload the three zip files from
`F:\MRI data\V1.6\` directly to:

```text
/path/to/aenet/data/m4raw/zips/
```

Keep the exact filenames. The download job will verify their size and MD5
before extraction.
