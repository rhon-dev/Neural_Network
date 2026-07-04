"""Phase 8 CNN training driver for MNIST.

Mirrors :mod:`src.train` but drives the :class:`src.cnn.CNN` architecture
instead of the MLP ``Network``. The extra step versus the MLP loop is a
reshape ``[N, 784] -> [N, 1, 28, 28]`` at the batch iterator boundary.

Exposes ``train_cnn(...)``, ``evaluate_cnn(net, X, y)``, and a
``python -m src.train_cnn`` CLI.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.cnn import CNN
from src.data.mnist import batch_iterator, load_mnist
from src.loss import cross_entropy_from_logits
from src.optimizers import Adam, SGD, SGDMomentum

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "mnist_cnn.npz"


def _build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return SGDMomentum(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"Unknown optimizer {name!r}")


def _reshape_images(X: np.ndarray) -> np.ndarray:
    """Turn ``[N, 784]`` into ``[N, 1, 28, 28]`` for the conv stack."""
    return X.reshape(X.shape[0], 1, 28, 28)


def evaluate_cnn(
    net: CNN, X: np.ndarray, y: np.ndarray, batch_size: int = 256
):
    """Compute mean loss and accuracy on a dataset in mini-batches.

    Args:
        net: trained ``CNN``.
        X: inputs of shape ``[N, 784]`` (will be reshaped internally).
        y: one-hot labels ``[N, 10]``.
    """
    N = X.shape[0]
    if N == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_correct = 0
    labels = np.argmax(y, axis=1)
    for start in range(0, N, batch_size):
        end = start + batch_size
        Xb = _reshape_images(X[start:end])
        yb = y[start:end]
        probs = net.forward(Xb)
        total_loss += cross_entropy_from_logits(
            net.logits_cache, yb, reduction="sum"
        )
        preds = np.argmax(probs, axis=1)
        total_correct += int(np.sum(preds == labels[start:end]))
    return total_loss / N, total_correct / N


def save_cnn_checkpoint(net: CNN, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        conv1_W=net.conv1.W, conv1_b=net.conv1.b,
        conv2_W=net.conv2.W, conv2_b=net.conv2.b,
        head_W=net.head.W,   head_b=net.head.b,
    )
    return path


def load_cnn_checkpoint(net: CNN, path: Path | str) -> None:
    data = np.load(path)
    net.conv1.W = data["conv1_W"].astype(np.float64, copy=True)
    net.conv1.b = data["conv1_b"].astype(np.float64, copy=True)
    net.conv2.W = data["conv2_W"].astype(np.float64, copy=True)
    net.conv2.b = data["conv2_b"].astype(np.float64, copy=True)
    net.head.W = data["head_W"].astype(np.float64, copy=True)
    net.head.b = data["head_b"].astype(np.float64, copy=True)


def train_cnn(
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    optimizer: str = "adam",
    seed: int = 0,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT_PATH,
    verbose: bool = True,
    train_subset: int | None = None,
    val_subset: int | None = None,
):
    """Train the CNN on MNIST. See :func:`src.train.train` for the analogous
    MLP function; the CLI arguments are the same.
    """
    (X_tr, y_tr), (X_val, y_val), _ = load_mnist(seed=seed)
    if train_subset is not None:
        X_tr, y_tr = X_tr[:train_subset], y_tr[:train_subset]
    if val_subset is not None:
        X_val, y_val = X_val[:val_subset], y_val[:val_subset]

    net = CNN(seed=seed)
    opt = _build_optimizer(optimizer, lr)

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for Xb_flat, yb in batch_iterator(
            X_tr, y_tr, batch_size, shuffle=True, seed=seed + epoch
        ):
            Xb = _reshape_images(Xb_flat)
            probs = net.forward(Xb)
            batch_loss = cross_entropy_from_logits(
                net.logits_cache, yb, reduction="mean"
            )
            net.backward(yb)
            opt.step(net.params_and_grads())

            n = Xb.shape[0]
            total_loss += batch_loss * n
            preds = np.argmax(probs, axis=1)
            total_correct += int(np.sum(preds == np.argmax(yb, axis=1)))
            total_samples += n

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_loss, val_acc = evaluate_cnn(net, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1:2d}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )

    if checkpoint_path is not None:
        path = save_cnn_checkpoint(net, checkpoint_path)
        if verbose:
            print(f"Saved checkpoint to {path}")

    return net, history


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the MNIST CNN.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--optimizer", choices=["sgd", "momentum", "adam"], default="adam"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH)
    )
    args = p.parse_args(argv)
    ckpt = args.checkpoint if args.checkpoint else None
    net, history = train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        checkpoint_path=ckpt,
    )
    # Test-set evaluation.
    (_tr, _val, (X_te, y_te)) = load_mnist(seed=args.seed)
    test_loss, test_acc = evaluate_cnn(net, X_te, y_te)
    print(f"Final val_acc  = {history['val_acc'][-1]:.4f}")
    print(f"Test loss      = {test_loss:.4f}")
    print(f"Test accuracy  = {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
