"""Tests for the Phase 6 training loop and its helpers.

The full MNIST run is not exercised by pytest (it takes tens of seconds and
the exit criterion is verified by a top-level script). These tests cover:

- ``_build_optimizer`` name lookup and validation.
- ``_iter_params_and_grads`` returns the right sequence of parameter tensors.
- ``evaluate`` returns valid loss/accuracy on synthetic data.
- ``save_checkpoint`` + ``load_checkpoint`` round-trip.
- Integration: a short training run on a small MNIST subset must actually
  reduce loss and beat random-guess accuracy.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src.network import Network
from src.train import (
    _build_optimizer,
    _iter_params_and_grads,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


def _mnist_available() -> bool:
    """Skip the integration test in environments without MNIST downloaded."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    required = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    return all((data_dir / name).exists() for name in required)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_returns_sgd(self):
        from src.optimizers import SGD

        assert isinstance(_build_optimizer("sgd", 0.01), SGD)

    def test_returns_momentum(self):
        from src.optimizers import SGDMomentum

        assert isinstance(_build_optimizer("momentum", 0.01), SGDMomentum)

    def test_returns_adam(self):
        from src.optimizers import Adam

        assert isinstance(_build_optimizer("adam", 1e-3), Adam)

    def test_case_insensitive(self):
        from src.optimizers import Adam

        assert isinstance(_build_optimizer("ADAM", 1e-3), Adam)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            _build_optimizer("rmsprop", 1e-3)


# ---------------------------------------------------------------------------
# Parameter iteration
# ---------------------------------------------------------------------------


class TestIterParamsAndGrads:
    def test_returns_six_pairs_for_three_layer_net(self):
        net = Network(seed=0)
        # Populate .dW / .db by running one forward+backward.
        X = np.zeros((2, 784))
        y = _one_hot(np.array([0, 1]), 10)
        net.forward(X)
        net.backward(y)

        pg = _iter_params_and_grads(net)
        assert len(pg) == 6  # 3 layers * (W, b)

        # Each pair references the actual layer arrays (identity, not a copy).
        for i, layer in enumerate(net.layers):
            W, dW = pg[2 * i]
            b, db = pg[2 * i + 1]
            assert W is layer.W
            assert b is layer.b
            assert dW is layer.dW
            assert db is layer.db


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_shapes_and_ranges(self):
        net = Network(seed=0)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 784))
        y = _one_hot(rng.integers(0, 10, size=20), 10)
        loss, acc = evaluate(net, X, y, batch_size=8)
        assert np.isfinite(loss)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_batched_agrees_with_single_pass(self):
        net = Network(seed=0)
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 784))
        y = _one_hot(rng.integers(0, 10, size=30), 10)
        loss_a, acc_a = evaluate(net, X, y, batch_size=8)
        loss_b, acc_b = evaluate(net, X, y, batch_size=30)
        np.testing.assert_allclose(loss_a, loss_b, rtol=1e-12)
        assert acc_a == acc_b

    def test_empty_input(self):
        net = Network(seed=0)
        X = np.zeros((0, 784))
        y = np.zeros((0, 10))
        loss, acc = evaluate(net, X, y)
        assert loss == 0.0 and acc == 0.0


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_round_trip_preserves_weights(self, tmp_path):
        net_a = Network(seed=0)
        # Run a forward+backward+step so weights are non-zero and biases too.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 784))
        y = _one_hot(rng.integers(0, 10, size=4), 10)
        net_a.forward(X)
        net_a.backward(y)
        for layer in net_a.layers:
            layer.b += 0.1  # break the initial-zero bias to prove round-trip.

        path = tmp_path / "ckpt.npz"
        save_checkpoint(net_a, path)
        assert path.exists()

        net_b = Network(seed=999)  # different init
        # Sanity: pre-load weights differ.
        assert not np.allclose(net_a.layers[0].W, net_b.layers[0].W)
        load_checkpoint(net_b, path)
        for la, lb in zip(net_a.layers, net_b.layers):
            np.testing.assert_allclose(la.W, lb.W)
            np.testing.assert_allclose(la.b, lb.b)

    def test_save_creates_parent_directories(self, tmp_path):
        net = Network(seed=0)
        deep_path = tmp_path / "a" / "b" / "c" / "ckpt.npz"
        save_checkpoint(net, deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Integration: short training run on real MNIST
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _mnist_available(),
    reason="MNIST not downloaded; skipping integration test.",
)
class TestTrainingLoopIntegration:
    def test_loss_decreases_on_small_subset(self, tmp_path, capsys):
        """Run 2 epochs on a small subset — loss should drop and val_acc
        should clear the random baseline."""
        _, history = train(
            epochs=2,
            batch_size=64,
            lr=1e-3,
            optimizer="adam",
            seed=0,
            checkpoint_path=tmp_path / "ckpt.npz",
            verbose=False,
            train_subset=2000,
            val_subset=1000,
        )
        assert len(history["train_loss"]) == 2
        assert history["train_loss"][1] < history["train_loss"][0]
        # Random-guess baseline is 0.1; a 2000-sample Adam warmup easily
        # beats 0.5 val_acc.
        assert history["val_acc"][-1] > 0.5
        # Checkpoint was written.
        assert (tmp_path / "ckpt.npz").exists()
