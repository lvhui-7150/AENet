"""Build a CIFAR-10 32x32 inpainting dataset from readable batch files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def read_batch(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    images = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    return images.astype(np.uint8), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, default=25000)
    parser.add_argument("--val-size", type=int, default=2500)
    parser.add_argument("--test-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images = []
    labels = []
    used_batches = []
    for path in sorted(input_dir.glob("data_batch_*")):
        try:
            batch_images, batch_labels = read_batch(path)
        except Exception as exc:
            print(f"[skip] {path.name}: {type(exc).__name__}: {exc}")
            continue
        images.append(batch_images)
        labels.append(batch_labels)
        used_batches.append(path.name)
        print(f"[read] {path.name}: {len(batch_images)} images")

    if not images:
        raise RuntimeError("no complete CIFAR batch files were found")

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    order = np.random.default_rng(args.seed).permutation(len(images))
    images = images[order]
    labels = labels[order]
    required = args.train_size + args.val_size + args.test_size
    if required > len(images):
        raise ValueError(
            f"requested {required} images but only {len(images)} are available"
        )

    train_end = args.train_size
    val_end = train_end + args.val_size
    test_end = val_end + args.test_size
    data = {
        "train_images": images[:train_end],
        "train_labels": labels[:train_end],
        "val_images": images[train_end:val_end],
        "val_labels": labels[train_end:val_end],
        "test_images": images[val_end:test_end],
        "test_labels": labels[val_end:test_end],
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "CIFAR-10",
                "image_size": 32,
                "channels": 3,
                "classes": DEFAULT_CLASSES,
                "used_batches": used_batches,
                "split_sizes": {
                    "train": args.train_size,
                    "val": args.val_size,
                    "test": args.test_size,
                },
                "seed": args.seed,
                "format": "uint8 NHWC",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] total={len(data['train_images']) + len(data['val_images']) + len(data['test_images'])}")
    print(f"[done] saved {output_path}")


if __name__ == "__main__":
    main()
