"""Prepare fastMRI knee data for the current AENet model zoo.

The output tensor has the same keys as the M4Raw prepared files:

    gt_complex  (N, H, W) complex64
    smap        (N, C, H, W) complex64
    kspace      (N, C, H, W) complex64
    masks       dict[str, (H, W) float32]
    split       dict[str, list[int]]

For single-coil knee data, C is one and the sensitivity map is all ones.
For multi-coil knee data, sensitivity maps are estimated from a low-frequency
calibration band and all volumes are padded to the largest coil count in the
selected subset.

The test split is carved out at the volume level from the official validation
directory because the public fastMRI test split does not release ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def fft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(x, axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c_np(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.ifftshift(x, axes=(-2, -1)),
        axes=(-2, -1),
        norm="ortho",
    )


def center_crop_or_pad(x: np.ndarray, size: int) -> np.ndarray:
    """Center crop or zero-pad the last two spatial dimensions."""
    height, width = x.shape[-2:]
    if height < size:
        top = (size - height) // 2
        bottom = size - height - top
        pad = [(0, 0)] * x.ndim
        pad[-2] = (top, bottom)
        x = np.pad(x, pad, mode="constant")
    if width < size:
        left = (size - width) // 2
        right = size - width - left
        pad = [(0, 0)] * x.ndim
        pad[-1] = (left, right)
        x = np.pad(x, pad, mode="constant")
    height, width = x.shape[-2:]
    top = (height - size) // 2
    left = (width - size) // 2
    return x[..., top : top + size, left : left + size]


def estimate_smaps(kspace: np.ndarray, calib_size: int) -> np.ndarray:
    """Estimate normalized coil sensitivity maps from low-frequency k-space."""
    _, coils, height, width = kspace.shape
    calib_size = min(calib_size, height, width)
    calibration = np.zeros((height, width), dtype=kspace.real.dtype)
    top = (height - calib_size) // 2
    left = (width - calib_size) // 2
    calibration[top : top + calib_size, left : left + calib_size] = 1.0
    low = ifft2c_np(kspace * calibration[None, None])
    denom = np.sqrt(np.sum(np.abs(low) ** 2, axis=1, keepdims=True))
    return (low / np.maximum(denom, 1e-12)).astype(np.complex64)


def fastmri_variable_density_mask(
    height: int,
    width: int,
    acceleration: int,
    center_fraction: float,
    seed: int,
) -> np.ndarray:
    """Create the exact-density 1D mask used by the earlier fastMRI pipeline.

    The total number of sampled phase-encode lines is round(width / R), so the
    effective acceleration is exactly R. The center band is fully sampled and
    the outer lines are selected as symmetric random pairs.
    """
    rng = np.random.default_rng(seed)
    num_center = max(1, int(round(width * center_fraction)))
    num_total = max(num_center, int(round(width / acceleration)))
    selected = np.zeros(width, dtype=bool)
    center_start = (width - num_center) // 2
    selected[center_start : center_start + num_center] = True

    center_set = set(np.flatnonzero(selected).tolist())
    pairs = [
        column
        for column in range(width // 2)
        if column not in center_set and (width - 1 - column) not in center_set
    ]
    num_outer = num_total - num_center
    num_pairs = min(num_outer // 2, len(pairs))
    if num_pairs:
        chosen = rng.choice(np.asarray(pairs, dtype=int), size=num_pairs, replace=False)
        selected[chosen] = True
        selected[width - 1 - chosen] = True
    if selected.sum() < num_total:
        free = np.flatnonzero(~selected)
        needed = min(num_total - int(selected.sum()), len(free))
        if needed:
            selected[rng.choice(free, size=needed, replace=False)] = True
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:, selected] = 1.0
    return mask


def list_h5_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*.h5") if path.is_file())
    if not paths:
        raise RuntimeError(f"no .h5 files found under {root}")
    return paths


def evenly_select(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(paths):
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in indices]


def split_validation_paths(
    paths: list[Path],
    num_val: int,
    num_test: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    np.random.default_rng(seed).shuffle(shuffled)
    if num_val <= 0:
        num_val = max(1, len(shuffled) // 2)
    if num_test <= 0:
        num_test = max(1, len(shuffled) - num_val)
    if num_val + num_test > len(shuffled):
        raise ValueError(
            f"requested {num_val} validation and {num_test} test volumes, "
            f"but only {len(shuffled)} validation volumes are available"
        )
    return shuffled[:num_val], shuffled[num_val : num_val + num_test]


def select_slice_indices(
    reference: np.ndarray,
    max_slices: int,
    min_rms_fraction: float,
) -> np.ndarray:
    """Select central, non-empty slices and spread them across each volume."""
    energy = np.mean(np.abs(reference) ** 2, axis=(-2, -1))
    finite = np.isfinite(energy)
    energy = np.where(finite, energy, 0.0)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        valid = np.arange(reference.shape[0], dtype=int)
    else:
        valid = np.flatnonzero(energy >= max(min_rms_fraction * peak, 1e-12))
    if valid.size == 0:
        valid = np.arange(reference.shape[0], dtype=int)
    if max_slices <= 0 or max_slices >= valid.size:
        return valid

    low = valid[int(0.10 * (len(valid) - 1))]
    high = valid[int(0.90 * (len(valid) - 1))]
    central = valid[(valid >= low) & (valid <= high)]
    if central.size == 0:
        central = valid
    positions = np.linspace(0, central.size - 1, max_slices).round().astype(int)
    return np.unique(central[positions])


def read_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        kspace = np.asarray(handle["kspace"][:], dtype=np.complex64)
        if "reconstruction_rss" in handle:
            reference = np.asarray(handle["reconstruction_rss"][:], dtype=np.float32)
        elif "reconstruction_esc" in handle:
            reference = np.asarray(handle["reconstruction_esc"][:], dtype=np.float32)
        else:
            if kspace.ndim == 4:
                reference = np.sqrt(
                    np.sum(np.abs(ifft2c_np(kspace)) ** 2, axis=1)
                ).astype(np.float32)
            else:
                reference = np.abs(ifft2c_np(kspace)).astype(np.float32)
    return kspace, reference


def process_singlecoil(
    kspace: np.ndarray,
    indices: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = kspace[indices]
    image = ifft2c_np(selected)
    if image_size > 0 and image.shape[-1] != image_size:
        image = center_crop_or_pad(image, image_size)
        selected = fft2c_np(image)
    scale = np.max(np.abs(image), axis=(-2, -1), keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    image = (image / scale).astype(np.complex64)
    selected = (selected / scale).astype(np.complex64)
    smap = np.ones_like(image)[:, None, :, :]
    return image, smap, selected


def process_multicoil(
    kspace: np.ndarray,
    indices: np.ndarray,
    image_size: int,
    calib_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = kspace[indices]
    smap = estimate_smaps(selected, calib_size)
    coil_images = ifft2c_np(selected)
    image = np.sum(np.conj(smap) * coil_images, axis=1)
    if image_size > 0 and image.shape[-1] != image_size:
        image = center_crop_or_pad(image, image_size)
        smap = center_crop_or_pad(smap, image_size)
        selected = fft2c_np(image[:, None, :, :] * smap)
    scale = np.max(np.abs(image), axis=(-2, -1), keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    image = (image / scale).astype(np.complex64)
    selected = (selected / scale[:, None, :, :]).astype(np.complex64)
    return image, smap.astype(np.complex64), selected


def process_split(
    paths: list[Path],
    split_name: str,
    modality: str,
    image_size: int,
    max_slices_per_volume: int,
    min_rms_fraction: float,
    calib_size: int,
):
    gt_parts: list[np.ndarray] = []
    smap_parts: list[np.ndarray] = []
    kspace_parts: list[np.ndarray] = []
    names: list[str] = []
    volume_meta = []

    for path in paths:
        kspace, reference = read_volume(path)
        indices = select_slice_indices(
            reference,
            max_slices_per_volume,
            min_rms_fraction,
        )
        if indices.size == 0:
            continue
        if modality == "singlecoil":
            if kspace.ndim != 3:
                raise ValueError(
                    f"singlecoil mode expects (S,H,W), got {kspace.shape} for {path}"
                )
            gt, smap, selected = process_singlecoil(
                kspace,
                indices,
                image_size,
            )
        else:
            if kspace.ndim != 4:
                raise ValueError(
                    f"multicoil mode expects (S,C,H,W), got {kspace.shape} for {path}"
                )
            gt, smap, selected = process_multicoil(
                kspace,
                indices,
                image_size,
                calib_size,
            )
        gt_parts.append(gt)
        smap_parts.append(smap)
        kspace_parts.append(selected)
        names.extend(f"{path.name}#{int(index)}" for index in indices)
        volume_meta.append(
            {
                "source": str(path),
                "slices": int(indices.size),
                "slice_indices": indices.astype(int).tolist(),
                "coils": int(smap.shape[1]),
            }
        )
        print(
            f"[prepare] {split_name} {path.name} "
            f"slices={indices.size} coils={smap.shape[1]}"
        )

    return gt_parts, smap_parts, kspace_parts, names, volume_meta


def pad_coils(
    smap_parts: list[np.ndarray],
    kspace_parts: list[np.ndarray],
    max_coils: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    padded_smap = []
    padded_kspace = []
    for smap, kspace in zip(smap_parts, kspace_parts):
        pad = max_coils - smap.shape[1]
        if pad <= 0:
            padded_smap.append(smap)
            padded_kspace.append(kspace)
            continue
        padding = ((0, 0), (0, pad), (0, 0), (0, 0))
        padded_smap.append(np.pad(smap, padding, mode="constant"))
        padded_kspace.append(np.pad(kspace, padding, mode="constant"))
    return padded_smap, padded_kspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        choices=("singlecoil", "multicoil"),
        default="singlecoil",
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="center crop/pad size; use 0 to keep the native fastMRI size",
    )
    parser.add_argument(
        "--train-files",
        type=int,
        default=200,
        help="0 uses every file in --train-dir",
    )
    parser.add_argument("--val-files", type=int, default=50)
    parser.add_argument("--test-files", type=int, default=50)
    parser.add_argument(
        "--max-slices-per-volume",
        type=int,
        default=4,
        help="0 keeps every valid slice in each selected volume",
    )
    parser.add_argument("--min-rms-fraction", type=float, default=0.05)
    parser.add_argument("--calib-size", type=int, default=32)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_paths = evenly_select(
        list_h5_files(Path(args.train_dir)),
        args.train_files,
    )
    validation_pool = list_h5_files(Path(args.val_dir))
    val_paths, test_paths = split_validation_paths(
        validation_pool,
        args.val_files,
        args.test_files,
        args.seed,
    )

    train_output = process_split(
        train_paths,
        "train",
        args.modality,
        args.image_size,
        args.max_slices_per_volume,
        args.min_rms_fraction,
        args.calib_size,
    )
    val_output = process_split(
        val_paths,
        "val",
        args.modality,
        args.image_size,
        args.max_slices_per_volume,
        args.min_rms_fraction,
        args.calib_size,
    )
    test_output = process_split(
        test_paths,
        "test",
        args.modality,
        args.image_size,
        args.max_slices_per_volume,
        args.min_rms_fraction,
        args.calib_size,
    )

    gt_parts = train_output[0] + val_output[0] + test_output[0]
    smap_parts = train_output[1] + val_output[1] + test_output[1]
    kspace_parts = train_output[2] + val_output[2] + test_output[2]
    names = train_output[3] + val_output[3] + test_output[3]
    if not gt_parts:
        raise RuntimeError("no slices were selected")

    max_coils = max(part.shape[1] for part in smap_parts)
    smap_parts, kspace_parts = pad_coils(smap_parts, kspace_parts, max_coils)

    num_train = len(train_output[0])
    num_val = len(val_output[0])
    split = {
        "train": list(range(0, num_train)),
        "val": list(range(num_train, num_train + num_val)),
        "test": list(
            range(num_train + num_val, num_train + num_val + len(test_output[0]))
        ),
    }
    height, width = gt_parts[0].shape[-2:]
    masks = {
        "r4_s42": torch.from_numpy(
            fastmri_variable_density_mask(
                height,
                width,
                4,
                args.center_fraction,
                args.seed,
            )
        ),
        "r8_s42": torch.from_numpy(
            fastmri_variable_density_mask(
                height,
                width,
                8,
                args.center_fraction,
                args.seed,
            )
        ),
    }
    data = {
        "gt_complex": torch.from_numpy(np.concatenate(gt_parts, axis=0)),
        "smap": torch.from_numpy(np.concatenate(smap_parts, axis=0)),
        "kspace": torch.from_numpy(np.concatenate(kspace_parts, axis=0)),
        "masks": masks,
        "split": split,
        "names": names,
        "meta": {
            "dataset": f"fastmri_knee_{args.modality}",
            "train": train_output[4],
            "val": val_output[4],
            "test": test_output[4],
        },
        "config": {
            **vars(args),
            "dataset": f"fastmri_knee_{args.modality}",
            "coils": int(max_coils),
        },
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    metadata = {
        "dataset": data["meta"]["dataset"],
        "samples": int(data["gt_complex"].shape[0]),
        "split_sizes": {key: len(value) for key, value in split.items()},
        "shape": list(data["gt_complex"].shape),
        "coils": int(max_coils),
        "mask_nlines": {
            key: int(value.sum().item()) for key, value in masks.items()
        },
        "config": data["config"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] samples={data['gt_complex'].shape[0]} "
        f"splits={ {key: len(value) for key, value in split.items()} } "
        f"coils={max_coils}"
    )
    print(f"[done] saved {output_path}")


if __name__ == "__main__":
    main()
