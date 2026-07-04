"""Tests for the Phase 7 evaluation module."""

from pathlib import Path

import numpy as np
import pytest

from src.evaluate import (
    confusion_matrix,
    evaluate_test_set,
    plot_confusion_matrix,
    plot_misclassifications,
    plot_training_curves,
)
from src.network import Network


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


class TestConfusionMatrix:
    def test_perfect_prediction_gives_diagonal(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        cm = confusion_matrix(y_true, y_pred, num_classes=5)
        np.testing.assert_array_equal(cm, np.eye(5, dtype=np.int64))

    def test_row_sums_equal_class_counts(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 10, size=200)
        y_pred = rng.integers(0, 10, size=200)
        cm = confusion_matrix(y_true, y_pred)
        for k in range(10):
            assert cm[k].sum() == int(np.sum(y_true == k))

    def test_total_equals_sample_count(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 10, size=137)
        y_pred = rng.integers(0, 10, size=137)
        cm = confusion_matrix(y_true, y_pred)
        assert cm.sum() == 137

    def test_off_diagonal_placement(self):
        # 3 samples of class 2 all mispredicted as class 5.
        y_true = np.array([2, 2, 2])
        y_pred = np.array([5, 5, 5])
        cm = confusion_matrix(y_true, y_pred, num_classes=10)
        assert cm[2, 5] == 3
        assert cm[2, 2] == 0
        assert cm.sum() == 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix(np.array([0, 1, 2]), np.array([0, 1]))


class TestEvaluateTestSet:
    def test_shapes_and_ranges(self):
        rng = np.random.default_rng(0)
        net = Network(seed=0)
        X = rng.normal(size=(30, 784))
        y = _one_hot(rng.integers(0, 10, size=30), 10)
        loss, acc, y_true_idx, y_pred_idx = evaluate_test_set(net, X, y)
        assert np.isfinite(loss)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0
        assert y_true_idx.shape == (30,)
        assert y_pred_idx.shape == (30,)
        assert y_true_idx.dtype == np.int64
        assert y_pred_idx.dtype == np.int64

    def test_accuracy_matches_predicted_indices(self):
        rng = np.random.default_rng(1)
        net = Network(seed=0)
        X = rng.normal(size=(50, 784))
        y = _one_hot(rng.integers(0, 10, size=50), 10)
        _, acc, y_true_idx, y_pred_idx = evaluate_test_set(net, X, y)
        expected = float(np.mean(y_true_idx == y_pred_idx))
        assert abs(acc - expected) < 1e-12


class TestPlotHelpers:
    def test_training_curves_creates_file(self, tmp_path):
        history = {
            "train_loss": [0.5, 0.3, 0.1],
            "val_loss": [0.6, 0.4, 0.2],
            "train_acc": [0.7, 0.85, 0.95],
            "val_acc": [0.65, 0.8, 0.9],
        }
        path = plot_training_curves(history, tmp_path / "curves.png")
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_confusion_matrix_creates_file(self, tmp_path):
        cm = np.eye(10, dtype=np.int64) * 100
        cm[0, 1] = 5
        path = plot_confusion_matrix(cm, tmp_path / "cm.png")
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_misclassifications_creates_file(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(30, 784))
        y_true = rng.integers(0, 10, size=30)
        y_pred = y_true.copy()
        # Make a few mispredictions.
        y_pred[:5] = (y_pred[:5] + 1) % 10
        path = plot_misclassifications(
            X, y_true, y_pred, tmp_path / "miscls.png", n=25
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    def test_misclassifications_handles_zero_wrong(self, tmp_path):
        # If nothing is wrong, we should still produce a file (an empty grid).
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=(10, 784))
        y = rng.integers(0, 10, size=10)
        path = plot_misclassifications(X, y, y, tmp_path / "miscls.png", n=25)
        assert Path(path).exists()
