"""Prepare a small M4Raw Brain multicoil dataset for AENet experiments.

M4Raw H5 files store:

    kspace             (slices, coils, H, W) complex64
    reconstruction_rss (slices, H, W)        float32

This script reads selected volumes straight from the Zenodo zip archives,
estimates coil sensitivity maps from a low-frequency calibration region, and
saves a torch ``.pt`` file that the multicoil AENet trainer can load directly.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch


def fft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(x, axes=(-2, -1), norm="ortho"), axes=(-2, -1)
    )


def ifft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"
    )


def cartesian_mask(h: int, w: int, accel: int, center: int, seed: int) -> np.ndarray:
    """Return a 1D Cartesian mask with a fixed fully sampled center band."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=np.float32)
    c0 = (w - center) // 2
    c1 = c0 + center
    center_idx = list(range(c0, c1))
    total = w // accel
    if total <= center:
        picked = sorted(rng.choice(center_idx, size=total, replace=False).tolist())
    else:
        n_outer = total - center
        offset = int(rng.integers(0, accel))
        outside = [
            j for j in range(w) if j not in set(center_idx) and (j - offset) % accel == 0
        ]
        outside = sorted(rng.choice(outside, size=n_outer, replace=False).tolist())
        picked = sorted(center_idx + outside)
    for j in picked:
        mask[:, j] = 1.0
    return mask


def estimate_smaps(kspace: np.ndarray, calib_size: int) -> np.ndarray:
    """Estimate normalized sensitivity maps from low-frequency k-space."""
    n_slices, coils, h, w = kspace.shape
    cal = np.zeros((h, w), dtype=kspace.real.dtype)
    c0 = (h - calib_size) // 2
    c1 = c0 + calib_size
    cal[c0:c1, c0:c1] = 1.0
    low = ifft2c_np(kspace * cal[None, None])
    denom = np.sqrt(np.sum(np.abs(low) ** 2, axis=1, keepdims=True))
    smap = low / np.maximum(denom, 1e-12)
    return smap.astype(np.complex64)


def read_volume(source_path: Path, entry_name: str, max_slices: int):
    """Read one H5 volume from an extracted directory or a zip archive."""
    def slice_indices(n_slices: int) -> np.ndarray:
        if max_slices <= 0 or max_slices >= n_slices:
            return np.arange(n_slices)
        return np.linspace(0, n_slices - 1, max_slices, dtype=int)

    if source_path.is_dir():
        with open(source_path / entry_name, "rb") as raw:
            with h5py.File(raw, "r") as f:
                idx = slice_indices(f["kspace"].shape[0])
                kspace = f["kspace"][idx]
                reconstruction_rss = f["reconstruction_rss"][idx]
        return kspace, reconstruction_rss
    else:
        with zipfile.ZipFile(source_path) as zf:
            raw = zf.read(entry_name)
        with h5py.File(io.BytesIO(raw), "r") as f:
            idx = slice_indices(f["kspace"].shape[0])
            kspace = f["kspace"][idx]
            reconstruction_rss = f["reconstruction_rss"][idx]
        return kspace, reconstruction_rss


def select_entries(source_path: Path, count: int) -> list[str]:
    if source_path.is_dir():
        entries = sorted(
            p.name for p in source_path.glob("*.h5")
        )
    else:
        with zipfile.ZipFile(source_path) as zf:
            entries = sorted(
                i.filename for i in zf.infolist() if i.filename.lower().endswith(".h5")
            )
    if count >= len(entries):
        return entries
    idx = np.linspace(0, len(entries) - 1, count, dtype=int).tolist()
    return [entries[i] for i in idx]


def process_volume(
    source_path: Path,
    entry_name: str,
    split_name: str,
    calib_size: int,
    max_slices: int,
):
    kspace, rss = read_volume(source_path, entry_name, max_slices)
    if kspace.shape[0] == 0:
        return [], [], [], []
    smaps = estimate_smaps(kspace, calib_size)
    coil_images = ifft2c_np(kspace)
    gt = np.sum(np.conj(smaps) * coil_images, axis=1)
    scale = rss.reshape(len(rss), -1).max(axis=1).clip(min=1e-6)
    scale = scale[:, None, None].astype(np.float32)
    gt = (gt / scale).astype(np.complex64)
    kspace = (kspace / scale[:, None, :, :]).astype(np.complex64)
    gt_list = [gt[i] for i in range(gt.shape[0])]
    kspace_list = [kspace[i] for i in range(kspace.shape[0])]
    smap_list = [smaps[i] for i in range(smaps.shape[0])]
    split_list = [split_name] * gt.shape[0]
    names = [f"{split_name}/{entry_name}#{i}" for i in range(gt.shape[0])]
    return gt_list, smap_list, kspace_list, split_list, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-zip",
        default=r"F:\MRI data\V1.6\M4RawV1.5_multicoil_train.zip",
        help="train source: zip archive or extracted directory of .h5 files",
    )
    ap.add_argument(
        "--val-zip",
        default=r"F:\MRI data\V1.6\M4RawV1.5_multicoil_val.zip",
        help="val source: zip archive or extracted directory of .h5 files",
    )
    ap.add_argument(
        "--test-zip",
        default=r"F:\MRI data\V1.6\M4Raw_multicoil_test.zip",
        help="test source: zip archive or extracted directory of .h5 files",
    )
    ap.add_argument(
        "--out",
        default=r"F:\MRI data\V1.6\prepared\m4raw_smoke.pt",
    )
    ap.add_argument("--train-files", type=int, default=4)
    ap.add_argument("--val-files", type=int, default=2)
    ap.add_argument("--test-files", type=int, default=2)
    ap.add_argument("--max-slices", type=int, default=18)
    ap.add_argument("--calib-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = [
        (Path(args.train_zip), args.train_files, "train"),
        (Path(args.val_zip), args.val_files, "val"),
        (Path(args.test_zip), args.test_files, "test"),
    ]

    all_gt: list[torch.Tensor] = []
    all_smap: list[torch.Tensor] = []
    all_kspace: list[torch.Tensor] = []
    split_lists: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    names: list[str] = []
    meta = {"volumes": []}

    for zip_path, count, split in specs:
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        entries = select_entries(zip_path, count)
        for entry in entries:
            gt, smap, kspace, _, _ = process_volume(
                zip_path, entry, split, args.calib_size, args.max_slices
            )
            if not gt:
                continue
            start = len(all_gt)
            all_gt.extend(torch.from_numpy(g) for g in gt)
            all_smap.extend(torch.from_numpy(s) for s in smap)
            all_kspace.extend(torch.from_numpy(k) for k in kspace)
            split_lists[split].extend(range(start, len(all_gt)))
            names.extend(f"{split}/{entry}#{i}" for i in range(len(gt)))
            meta["volumes"].append(
                {
                    "split": split,
                    "entry": entry,
                    "slices": len(gt),
                    "start": start,
                    "end": len(all_gt),
                }
            )
            print(
                f"[prepare] {split} {entry} slices={len(gt)} "
                f"running_total={len(all_gt)}"
            )

    h, w = all_gt[0].shape[-2:]
    masks = {
        "r4_s42": torch.from_numpy(cartesian_mask(h, w, 4, 32, args.seed)),
        "r8_s42": torch.from_numpy(cartesian_mask(h, w, 8, 16, args.seed)),
    }
    data = {
        "gt_complex": torch.stack(all_gt),
        "smap": torch.stack(all_smap),
        "kspace": torch.stack(all_kspace),
        "masks": masks,
        "split": split_lists,
        "names": names,
        "meta": meta,
        "config": vars(args),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "samples": len(all_gt),
                "split_sizes": {k: len(v) for k, v in split_lists.items()},
                "shape": list(data["gt_complex"].shape),
                "mask_nlines": {k: int(v.sum().item()) for k, v in masks.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] samples={len(all_gt)} splits="
        f"{ {k: len(v) for k, v in split_lists.items()} }"
    )
    print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
