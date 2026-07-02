"""MNIST download, IDX parsing, normalization, one-hot encoding, splits, batching.

Phase 0 data pipeline. Raw files are downloaded into data/ (gitignored).
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download_mnist(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Download MNIST gzip files into data_dir if not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, fname in FILES.items():
        dest = data_dir / fname
        if not dest.exists():
            urllib.request.urlretrieve(MIRROR + fname, dest)
        paths[key] = dest
    return paths


def parse_idx_images(path: Path | str) -> np.ndarray:
    """Parse an IDX3 gzip file into a [N, rows*cols] uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic {magic} in {path}, expected 2051")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def parse_idx_labels(path: Path | str) -> np.ndarray:
    """Parse an IDX1 gzip file into a [N] uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic {magic} in {path}, expected 2049")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels [0, 255] to float64 [0, 1]."""
    return X.astype(np.float64) / 255.0


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Encode integer labels as [N, num_classes] one-hot float64 array."""
    labels = np.asarray(labels)
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_mnist(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    val_size: int = 10_000,
    seed: int = 0,
):
    """Load MNIST as normalized, one-hot-encoded train/val/test splits.

    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test) where
    X is [N, 784] float64 in [0, 1] and y is [N, 10] one-hot float64.
    The validation set is carved deterministically (seed) from the 60k
    training set; test is the official 10k test set.
    """
    paths = download_mnist(data_dir)

    X_full = normalize_pixels(parse_idx_images(paths["train_images"]))
    y_full = one_hot(parse_idx_labels(paths["train_labels"]))
    X_test = normalize_pixels(parse_idx_images(paths["test_images"]))
    y_test = one_hot(parse_idx_labels(paths["test_labels"]))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_full.shape[0])
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    return (
        (X_full[train_idx], y_full[train_idx]),
        (X_full[val_idx], y_full[val_idx]),
        (X_test, y_test),
    )


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    """Yield (X_batch, y_batch) mini-batches; last batch may be smaller."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, X.shape[0], batch_size):
        sel = idx[start : start + batch_size]
        yield X[sel], y[sel]
