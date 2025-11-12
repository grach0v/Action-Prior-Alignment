#!/usr/bin/env python3
"""
Utility to split the A2 preprocessed dataset into train/test partitions.

The dataset stored in `data/a2_pp_data.npy` is a pickled dict with several
fields (sequence, lang_goal, pts, etc.). This script shuffles the episodes,
splits them according to the requested ratio, and persists the result as two
NumPy dict files that keep the original structure so the existing data loaders
can consume them without changes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data.npy"),
        help="Path to the original dataset .npy file.",
    )
    parser.add_argument(
        "--train-output",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data_train.npy"),
        help="Path where the train split will be stored.",
    )
    parser.add_argument(
        "--test-output",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data_test.npy"),
        help="Path where the test split will be stored.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples to allocate to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling prior to splitting.",
    )
    parser.add_argument(
        "--metadata-output",
        type=pathlib.Path,
        default=pathlib.Path("data/a2_pp_data_split.json"),
        help="Optional JSON file that stores split metadata.",
    )
    return parser.parse_args()


def load_dataset(path: pathlib.Path) -> Dict[str, Sequence]:
    loaded = np.load(str(path), allow_pickle=True).item()
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dictionary payload, received {type(loaded)}")
    return loaded


def split_indices(count: int, test_ratio: float, seed: int) -> Dict[str, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)

    test_size = int(round(count * test_ratio))
    test_size = max(1, min(count - 1, test_size))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return {"train": np.sort(train_idx), "test": np.sort(test_idx)}


def subset_dict(dataset: Dict[str, Sequence], idx: np.ndarray) -> Dict[str, List]:
    return {key: [dataset[key][i] for i in idx] for key in dataset.keys()}


def save_split(path: pathlib.Path, payload: Dict[str, Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), payload, allow_pickle=True)


def save_metadata(
    path: pathlib.Path,
    *,
    input_path: pathlib.Path,
    train_path: pathlib.Path,
    test_path: pathlib.Path,
    test_ratio: float,
    seed: int,
    train_count: int,
    test_count: int,
) -> None:
    payload = {
        "input_path": str(input_path),
        "train_output": str(train_path),
        "test_output": str(test_path),
        "test_ratio": test_ratio,
        "seed": seed,
        "train_count": train_count,
        "test_count": test_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.input)
    total_samples = len(next(iter(dataset.values())))
    if total_samples == 0:
        raise ValueError("Cannot split an empty dataset.")

    split_idx = split_indices(total_samples, args.test_ratio, args.seed)
    train_split = subset_dict(dataset, split_idx["train"])
    test_split = subset_dict(dataset, split_idx["test"])

    save_split(args.train_output, train_split)
    save_split(args.test_output, test_split)
    save_metadata(
        args.metadata_output,
        input_path=args.input,
        train_path=args.train_output,
        test_path=args.test_output,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_count=len(split_idx["train"]),
        test_count=len(split_idx["test"]),
    )

    print(
        f"Split {total_samples} samples into "
        f"{len(split_idx['train'])} train / {len(split_idx['test'])} test"
    )


if __name__ == "__main__":
    main()

