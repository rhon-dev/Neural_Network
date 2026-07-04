"""Phase 6 training loop for the 784-128-64-10 MNIST MLP.

Integrates the modules built in earlier phases:

    Phase 0 — src/data/mnist.py       load_mnist, batch_iterator
    Phase 2 — src/network.py          forward pass
    Phase 3 — src/loss.py             cross_entropy_from_logits
    Phase 4 — src/network.py          backward pass
    Phase 5 — src/optimizers.py       SGD / SGDMomentum / Adam

Exposes:

- ``train(...)``: an epoch/batch loop with per-epoch validation and optional
  checkpointing. Returns the trained network and a history dict of per-epoch
  train/validation loss and accuracy.
- ``evaluate(net, X, y, batch_size)``: memory-safe batched evaluation.
- ``save_checkpoint`` / ``load_checkpoint``: NumPy ``.npz`` weight I/O.
- Command-line entry point: ``python -m src.train``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.data.mnist import batch_iterator, load_mnist
from src.loss import cross_entropy_from_logits
from src.network import Network
from src.optimizers import SGD, Adam, SGDMomentum

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "mnist_mlp.npz"


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


def _build_optimizer(name: str, lr: float):
    """Look up an optimizer by name and instantiate it with the given lr."""
    name = name.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return SGDMomentum(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(
        f"Unknown optimizer {name!r}; expected 'sgd', 'momentum', or 'adam'."
    )


def _iter_params_and_grads(net: Network):
    """Return the list of (param, grad) pairs Network exposes to an optimizer.

    Order matters — momentum and Adam key their state by list index, so the
    same call site must return the same ordering across training steps.
    """
    pg = []
    for layer in net.layers:
        pg.append((layer.W, layer.dW))
        pg.append((layer.b, layer.db))
    return pg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(net: Network, X: np.ndarray, y: np.ndarray, batch_size: int = 512):
    """Compute mean cross-entropy loss and classification accuracy.

    Runs the forward pass in mini-batches to keep peak memory bounded.

    Args:
        net: a ``Network`` in evaluation-ready state.
        X: inputs of shape ``[N, 784]``.
        y: one-hot labels of shape ``[N, 10]``.
        batch_size: batch size for the forward pass. Default: 512.

    Returns:
        ``(mean_loss, accuracy)`` — accuracy is in ``[0, 1]``.
    """
    N = X.shape[0]
    if N == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_correct = 0
    labels = np.argmax(y, axis=1)

    for start in range(0, N, batch_size):
        end = start + batch_size
        Xb = X[start:end]
        yb = y[start:end]
        probs = net.forward(Xb)
        # ``sum`` so we can average across batches on the same denominator.
        total_loss += cross_entropy_from_logits(
            net.logits_cache, yb, reduction="sum"
        )
        preds = np.argmax(probs, axis=1)
        total_correct += int(np.sum(preds == labels[start:end]))

    return total_loss / N, total_correct / N


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(net: Network, path: Path | str) -> Path:
    """Persist all layer weights and biases to a ``.npz`` archive.

    Args:
        net: a ``Network`` whose layers hold arrays ``W`` and ``b``.
        path: destination file path. Parent directories are created as needed.

    Returns:
        The final ``Path`` written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, layer in enumerate(net.layers):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    np.savez(path, **arrays)
    return path


def load_checkpoint(net: Network, path: Path | str) -> None:
    """Restore weights and biases from a ``save_checkpoint`` archive in place."""
    data = np.load(path)
    for i, layer in enumerate(net.layers):
        layer.W = data[f"W{i}"].astype(np.float64, copy=True)
        layer.b = data[f"b{i}"].astype(np.float64, copy=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    optimizer: str = "adam",
    seed: int = 0,
    checkpoint_path: Path | str | None = DEFAULT_CHECKPOINT_PATH,
    verbose: bool = True,
    data_dir: Path | str | None = None,
    train_subset: int | None = None,
    val_subset: int | None = None,
):
    """Train the fixed-architecture MLP on MNIST.

    Args:
        epochs: number of full passes over the training set.
        batch_size: mini-batch size.
        lr: learning rate.
        optimizer: one of ``"sgd"``, ``"momentum"``, ``"adam"``.
        seed: seed for network initialization, batch shuffling, and the
            train/val split.
        checkpoint_path: path to write the final ``.npz`` checkpoint. Set to
            ``None`` to skip saving.
        verbose: if ``True``, print a one-line summary per epoch.
        data_dir: optional override for the MNIST data directory.
        train_subset: if set, use only this many training examples (useful
            for fast tests). Applied after the train/val split.
        val_subset: if set, use only this many validation examples.

    Returns:
        ``(net, history)`` — the trained ``Network`` and a dict with keys
        ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``, each mapping
        to a list of per-epoch values.
    """
    load_kwargs = {"seed": seed}
    if data_dir is not None:
        load_kwargs["data_dir"] = data_dir
    (X_tr, y_tr), (X_val, y_val), _ = load_mnist(**load_kwargs)

    if train_subset is not None:
        X_tr, y_tr = X_tr[:train_subset], y_tr[:train_subset]
    if val_subset is not None:
        X_val, y_val = X_val[:val_subset], y_val[:val_subset]

    net = Network(seed=seed)
    opt = _build_optimizer(optimizer, lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    train_labels = np.argmax(y_tr, axis=1)

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for Xb, yb in batch_iterator(
            X_tr, y_tr, batch_size, shuffle=True, seed=seed + epoch
        ):
            probs = net.forward(Xb)
            batch_loss = cross_entropy_from_logits(
                net.logits_cache, yb, reduction="mean"
            )
            net.backward(yb)
            opt.step(_iter_params_and_grads(net))

            n = Xb.shape[0]
            total_loss += batch_loss * n
            preds = np.argmax(probs, axis=1)
            total_correct += int(np.sum(preds == np.argmax(yb, axis=1)))
            total_samples += n

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_loss, val_acc = evaluate(net, X_val, y_val)

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
        path = save_checkpoint(net, checkpoint_path)
        if verbose:
            print(f"Saved checkpoint to {path}")

    # Silence the reference to train_labels — retained for readability of the
    # loop above but not needed after the fact.
    del train_labels

    return net, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the MNIST MLP.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--optimizer", choices=["sgd", "momentum", "adam"], default="adam"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Where to write the final .npz checkpoint. Empty string disables.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    ckpt = args.checkpoint if args.checkpoint else None
    _, history = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        checkpoint_path=ckpt,
    )
    print(f"Final val_acc = {history['val_acc'][-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
