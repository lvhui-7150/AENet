"""Convert existing processed fastMRI knee tensors to the model-zoo format.

Supported inputs:

1. 128 x 128 legacy tensor:
       gt_complex (N, 128, 128)
       masks      dict[str, (128, 128)]
       split      dict with train/val/test lists

2. 320 x 320 chunked tensor:
       fastmri_320_meta.pt
       fastmri_320_gt_chunk_000.pt ... 007.pt

The converter adds the fields required by train_aenet_m4raw.py:

    smap    (N, 1, H, W)
    kspace  (N, 1, H, W)

The k-space is constructed from the image-domain ground truth using the same
centered orthonormal FFT convention as the training script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))


def evenly_select(indices: list[int], count: int) -> list[int]:
    if count <= 0 or count >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count)
    return [indices[int(round(position))] for position in positions]


def normalize_split(split: dict, num_slices: int) -> dict[str, list[int]]:
    normalized = {}
    for name in ("train", "val", "test"):
        values = [int(value) for value in split[name]]
        if values and (min(values) < 0 or max(values) >= num_slices):
            raise ValueError(
                f"{name} split contains an index outside [0, {num_slices})"
            )
        normalized[name] = values
    return normalized


def load_128(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    gt = data["gt_complex"]
    masks = {
        key: value.float()
        for key, value in data["masks"].items()
    }
    split = normalize_split(data["split"], len(gt))
    meta = data.get("meta", {"source": "fastmri_128_prepared"})
    return gt, masks, split, meta


def build_file_to_slices(slice_files: torch.Tensor) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for global_index, file_index in enumerate(slice_files.tolist()):
        mapping.setdefault(int(file_index), []).append(global_index)
    return mapping


def load_320(meta_path: Path, max_train: int, max_val: int, max_test: int):
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    file_to_slices = build_file_to_slices(meta["slice_files"])
    split_by_file = meta["split"]
    split = {}
    for name, key in (
        ("train", "train_files"),
        ("val", "val_files"),
        ("test", "test_files"),
    ):
        indices = [
            slice_index
            for file_index in split_by_file[key]
            for slice_index in file_to_slices[int(file_index)]
        ]
        limit = {"train": max_train, "val": max_val, "test": max_test}[name]
        split[name] = evenly_select(indices, limit)

    masks = {
        key: value["mask"].float()
        for key, value in meta["masks"].items()
    }
    selected = sorted(set(split["train"] + split["val"] + split["test"]))
    selected_by_chunk: dict[int, list[int]] = {}
    for global_index in selected:
        chunk_index = global_index // meta["gt"]["chunk_size_rows"]
        selected_by_chunk.setdefault(chunk_index, []).append(global_index)

    chunks = []
    legacy_dir = meta_path.parent
    for chunk_index in sorted(selected_by_chunk):
        chunk_meta = meta["gt"]["chunks"][chunk_index]
        chunk_path = legacy_dir / chunk_meta["name"]
        print(f"[convert] loading {chunk_path.name}")
        try:
            chunk = torch.load(
                chunk_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            chunk = torch.load(
                chunk_path,
                map_location="cpu",
                weights_only=False,
            )
        start = int(chunk_meta["start"])
        local_indices = [
            global_index - start for global_index in selected_by_chunk[chunk_index]
        ]
        chunks.append(chunk["gt"][local_indices])
    gt = torch.cat(chunks, dim=0)
    if len(gt) != len(selected):
        raise RuntimeError(f"loaded {len(gt)} slices but selected {len(selected)}")

    remap = {old: new for new, old in enumerate(selected)}
    split = {name: [remap[index] for index in values] for name, values in split.items()}
    meta_out = {
        "source": "fastmri_320_meta",
        "n_slices_available": int(meta["gt"]["n_slices"]),
        "n_slices_exported": len(gt),
    }
    return gt, masks, split, meta_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("128", "320"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.kind == "128":
        gt, masks, split, source_meta = load_128(input_path)
    else:
        gt, masks, split, source_meta = load_320(
            input_path,
            args.max_train,
            args.max_val,
            args.max_test,
        )

    gt = gt.contiguous()
    kspace = fft2c(gt).unsqueeze(1).contiguous()
    smap = torch.ones_like(gt).unsqueeze(1).contiguous()
    height, width = gt.shape[-2:]
    data = {
        "gt_complex": gt,
        "smap": smap,
        "kspace": kspace,
        "masks": masks,
        "split": split,
        "names": [f"fastmri_knee/{index}" for index in range(len(gt))],
        "meta": {
            "dataset": "fastmri_knee_singlecoil",
            "source": source_meta,
            "conversion": "gt -> centered orthonormal FFT -> image-domain kspace",
        },
        "config": {
            "dataset": "fastmri_knee_singlecoil",
            "coils": 1,
            "image_size": height if height == width else [height, width],
            "source_file": str(input_path),
        },
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": "fastmri_knee_singlecoil",
                "samples": int(gt.shape[0]),
                "split_sizes": {key: len(value) for key, value in split.items()},
                "shape": list(gt.shape),
                "masks": {key: int(value.sum()) for key, value in masks.items()},
                "source": source_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] samples={gt.shape[0]} "
        f"splits={ {key: len(value) for key, value in split.items()} }"
    )
    print(f"[done] saved {output_path}")


if __name__ == "__main__":
    main()
