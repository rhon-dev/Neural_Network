"""Phase 7 evaluation, visualization, and reporting.

This module owns the held-out test-set evaluation and every figure in
``docs/``. Nothing here modifies training-time code — the training loop
is imported and called from :func:`src.train.train`, and the trained
network's parameters are consumed via the checkpoint saved by Phase 6.

Public surface:

- :func:`evaluate_test_set` — mean loss, accuracy, and predicted-vs-true
  label vectors on the MNIST test split.
- :func:`confusion_matrix` — ``[K, K]`` counts array, rows = true class.
- :func:`plot_training_curves`, :func:`plot_confusion_matrix`,
  :func:`plot_misclassifications` — matplotlib figures, saved to disk.
- :func:`main` (CLI: ``python -m src.evaluate``) — end-to-end pipeline:
  train, evaluate, save history + figures, print the numbers that go
  into the README Results section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Non-interactive backend so this works over SSH / in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend switch)
import numpy as np  # noqa: E402

from src.data.mnist import load_mnist  # noqa: E402
from src.network import Network  # noqa: E402
from src.train import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    evaluate,
    load_checkpoint,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "docs"
DEFAULT_HISTORY_PATH = DEFAULT_FIGURE_DIR / "training_history.json"

NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------


def evaluate_test_set(net: Network, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate ``net`` on the held-out test split.

    Args:
        net: a trained ``Network``.
        X_test: test inputs of shape ``[N, 784]``.
        y_test: one-hot test labels of shape ``[N, 10]``.

    Returns:
        Tuple ``(test_loss, test_acc, y_true_idx, y_pred_idx)`` — the mean
        cross-entropy loss, top-1 accuracy in ``[0, 1]``, and integer
        class-index arrays of length ``N`` for the true and predicted
        labels respectively.
    """
    test_loss, test_acc = evaluate(net, X_test, y_test)

    # Recover integer predictions via a batched forward pass.
    batch_size = 512
    y_pred_idx = np.empty(X_test.shape[0], dtype=np.int64)
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        probs = net.forward(X_test[start:end])
        y_pred_idx[start:end] = np.argmax(probs, axis=1)

    y_true_idx = np.argmax(y_test, axis=1).astype(np.int64)
    return test_loss, test_acc, y_true_idx, y_pred_idx


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Build a ``[num_classes, num_classes]`` confusion matrix.

    Rows index the true class, columns the predicted class. ``cm[i, j]``
    is the number of samples whose true label is ``i`` and whose
    predicted label is ``j``.
    """
    y_true_idx = np.asarray(y_true_idx, dtype=np.int64)
    y_pred_idx = np.asarray(y_pred_idx, dtype=np.int64)
    if y_true_idx.shape != y_pred_idx.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_idx.shape} vs y_pred {y_pred_idx.shape}"
        )

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # np.add.at gives correct scattered counting even with duplicate indices.
    np.add.at(cm, (y_true_idx, y_pred_idx), 1)
    return cm


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict, out_path: Path | str) -> Path:
    """Two-panel plot: training + validation loss (left), accuracy (right).

    Args:
        history: dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc``, each a list of per-epoch values.
        out_path: destination image path.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, history["train_loss"], label="train", marker="o")
    ax_loss.plot(epochs, history["val_loss"], label="validation", marker="o")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train", marker="o")
    ax_acc.plot(epochs, history["val_acc"], label="validation", marker="o")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("MNIST training — 784-128-64-10 MLP, Adam")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: np.ndarray, out_path: Path | str) -> Path:
    """Plot a confusion matrix with per-cell counts annotated."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test-set confusion matrix")

    # Annotate cells with counts; use a light color on dark cells for contrast.
    vmax = cm.max() if cm.size else 1
    threshold = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", color=color, fontsize=8,
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_misclassifications(
    X: np.ndarray,
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    out_path: Path | str,
    n: int = 25,
) -> Path:
    """Save a grid of the first ``n`` misclassified test images.

    Args:
        X: test inputs of shape ``[N, 784]`` with entries in ``[0, 1]``.
        y_true_idx: integer true labels, length ``N``.
        y_pred_idx: integer predicted labels, length ``N``.
        out_path: destination image path.
        n: maximum number of misclassifications to show (grid is 5x5 by default).
    """
    wrong = np.where(y_true_idx != y_pred_idx)[0]
    take = wrong[:n]
    grid = int(np.ceil(np.sqrt(max(len(take), 1))))

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.6, grid * 1.7))
    axes = np.atleast_2d(axes)
    for ax in axes.flatten():
        ax.axis("off")

    for k, idx in enumerate(take):
        ax = axes.flatten()[k]
        img = X[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray_r")
        ax.set_title(
            f"true {y_true_idx[idx]}  pred {y_pred_idx[idx]}", fontsize=8
        )

    fig.suptitle(
        f"Representative misclassifications ({len(take)} of {len(wrong)} total)"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _save_history_json(history: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all values are JSON-serializable floats.
    serializable = {k: [float(v) for v in vs] for k, vs in history.items()}
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)
    return path


def _load_history_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 7 pipeline end to end and print the report."""
    parser = argparse.ArgumentParser(description="Evaluate the MNIST MLP.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Train from scratch even if a checkpoint already exists.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory to write figures into.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Where to persist per-epoch history JSON.",
    )
    args = parser.parse_args(argv)

    checkpoint_path = Path(DEFAULT_CHECKPOINT_PATH)
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    net = Network(seed=args.seed)

    need_train = args.retrain or not checkpoint_path.exists() or not args.history_path.exists()

    if need_train:
        print("Training from scratch...")
        net, history = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer=args.optimizer,
            seed=args.seed,
            checkpoint_path=checkpoint_path,
            verbose=True,
        )
        _save_history_json(history, args.history_path)
        print(f"Saved history to {args.history_path}")
    else:
        print(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(net, checkpoint_path)
        history = _load_history_json(args.history_path)

    # Test-set evaluation on the untouched official 10k split.
    (_tr, _val, (X_test, y_test)) = load_mnist(seed=args.seed)
    test_loss, test_acc, y_true_idx, y_pred_idx = evaluate_test_set(
        net, X_test, y_test
    )

    cm = confusion_matrix(y_true_idx, y_pred_idx)

    curves_path = plot_training_curves(history, figure_dir / "training_curves.png")
    cm_path = plot_confusion_matrix(cm, figure_dir / "confusion_matrix.png")
    miscls_path = plot_misclassifications(
        X_test, y_true_idx, y_pred_idx, figure_dir / "misclassifications.png"
    )

    n_wrong = int(np.sum(y_true_idx != y_pred_idx))
    print()
    print("=" * 60)
    print("Phase 7 — held-out test evaluation")
    print("=" * 60)
    print(f"Test loss           : {test_loss:.4f}")
    print(f"Test accuracy       : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print(f"Misclassified       : {n_wrong} of {len(y_true_idx)}")
    print(f"Final training loss : {history['train_loss'][-1]:.4f}")
    print(f"Final val accuracy  : {history['val_acc'][-1]:.4f}")
    print(f"Epochs trained      : {len(history['train_loss'])}")
    print()
    print("Figures:")
    print(f"  - {curves_path.relative_to(PROJECT_ROOT)}")
    print(f"  - {cm_path.relative_to(PROJECT_ROOT)}")
    print(f"  - {miscls_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
