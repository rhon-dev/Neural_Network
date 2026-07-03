"""Tests for categorical cross-entropy loss and its analytic gradient (Phase 3)."""

import numpy as np
import pytest

from src.activations import softmax
from src.gradient_check import check_gradient, relative_error
from src.loss import (
    cross_entropy,
    cross_entropy_from_logits,
    cross_entropy_grad_from_logits,
)


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


class TestCrossEntropyBasicProperties:
    """Sanity checks: non-negativity, boundary cases, shape handling."""

    def test_non_negative_on_random_inputs(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(16, 10))
        y = _one_hot(rng.integers(0, 10, size=16), 10)
        assert cross_entropy_from_logits(logits, y) >= 0.0
        assert cross_entropy(softmax(logits), y) >= 0.0

    def test_confident_correct_near_zero(self, capsys):
        # A logit vector with one very large entry maps to a near-one-hot
        # probability; the correct class should give a near-zero loss.
        logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        y = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        loss = cross_entropy_from_logits(logits, y)
        print(f"confident-correct loss = {loss:.6e}")
        assert loss < 1e-3

    def test_confident_wrong_is_large(self, capsys):
        # Same logits as above but the label is on a different class:
        # loss should be roughly (large logit) - log-sum-exp ~= large.
        logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0]])
        y = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        loss = cross_entropy_from_logits(logits, y)
        print(f"confident-wrong loss   = {loss:.6e}")
        assert loss > 5.0

    def test_uniform_prediction_equals_log_k(self):
        # Uniform logits => uniform softmax => loss = log(K) regardless of label.
        K = 10
        logits = np.zeros((4, K))
        y = _one_hot(np.array([0, 3, 7, 9]), K)
        loss = cross_entropy_from_logits(logits, y)
        np.testing.assert_allclose(loss, np.log(K), rtol=1e-12)

    def test_probability_input_one_hot_perfect_is_zero(self):
        # Feed in exact one-hot probabilities: with eps clipping the log is
        # log(1) = 0 on the correct class and 0 * log(eps) = 0 on the rest,
        # so the total loss is exactly 0.
        probs = np.array([[1.0, 0.0, 0.0, 0.0]])
        y = np.array([[1.0, 0.0, 0.0, 0.0]])
        loss = cross_entropy(probs, y)
        np.testing.assert_allclose(loss, 0.0, atol=1e-12)

    def test_1d_input_accepted(self):
        logits = np.array([2.0, 1.0, 0.1])
        y = np.array([1.0, 0.0, 0.0])
        loss = cross_entropy_from_logits(logits, y)
        assert np.isscalar(loss) or loss.shape == ()

    def test_shape_mismatch_raises(self):
        logits = np.zeros((3, 4))
        y = np.zeros((3, 5))
        with pytest.raises(ValueError):
            cross_entropy_from_logits(logits, y)


class TestCrossEntropyReduction:
    """Mean / sum / none reduction semantics."""

    def test_mean_equals_sum_over_n(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(8, 6))
        y = _one_hot(rng.integers(0, 6, size=8), 6)
        mean = cross_entropy_from_logits(logits, y, reduction="mean")
        total = cross_entropy_from_logits(logits, y, reduction="sum")
        np.testing.assert_allclose(mean, total / 8.0, rtol=1e-12)

    def test_none_returns_per_sample_vector(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(5, 4))
        y = _one_hot(rng.integers(0, 4, size=5), 4)
        per_sample = cross_entropy_from_logits(logits, y, reduction="none")
        assert per_sample.shape == (5,)
        assert np.all(per_sample >= 0.0)

    def test_unknown_reduction_raises(self):
        logits = np.zeros((2, 3))
        y = _one_hot(np.array([0, 1]), 3)
        with pytest.raises(ValueError):
            cross_entropy_from_logits(logits, y, reduction="weird")


class TestCrossEntropyLogitsVsProbs:
    """The from-logits and from-probs paths should agree on well-scaled inputs."""

    def test_paths_agree_on_moderate_logits(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(6, 10))
        y = _one_hot(rng.integers(0, 10, size=6), 10)
        loss_logits = cross_entropy_from_logits(logits, y)
        loss_probs = cross_entropy(softmax(logits), y)
        np.testing.assert_allclose(loss_logits, loss_probs, rtol=1e-10, atol=1e-12)


class TestCrossEntropyStability:
    """Numerical stability: no overflow / NaN on extreme logits."""

    def test_from_logits_no_overflow(self):
        logits = np.array([[10000.0, 10001.0, 9999.0]])
        y = np.array([[0.0, 1.0, 0.0]])
        loss = cross_entropy_from_logits(logits, y)
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_from_probs_eps_clip_survives_zero(self):
        # A zero probability on the true class would blow up under plain log,
        # but epsilon clipping keeps the loss finite (and large).
        probs = np.array([[0.0, 1.0]])
        y = np.array([[1.0, 0.0]])
        loss = cross_entropy(probs, y)
        assert np.isfinite(loss)
        assert loss > 20.0


class TestCrossEntropyGradient:
    """Analytic gradient matches numerical gradient via the Phase 1 utility."""

    def test_gradient_matches_softmax_minus_y(self):
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(3, 5))
        y = _one_hot(rng.integers(0, 5, size=3), 5)
        expected = (softmax(logits) - y) / 3.0
        analytic = cross_entropy_grad_from_logits(logits, y, reduction="mean")
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_gradient_sum_reduction_matches_softmax_minus_y(self):
        rng = np.random.default_rng(5)
        logits = rng.normal(size=(3, 5))
        y = _one_hot(rng.integers(0, 5, size=3), 5)
        expected = softmax(logits) - y
        analytic = cross_entropy_grad_from_logits(logits, y, reduction="sum")
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_gradcheck_passes_mean_reduction(self, capsys):
        rng = np.random.default_rng(6)
        logits = rng.normal(size=(4, 6))
        y = _one_hot(rng.integers(0, 6, size=4), 6)

        def loss_fn(z):
            return cross_entropy_from_logits(z, y, reduction="mean")

        def grad_fn(z):
            return cross_entropy_grad_from_logits(z, y, reduction="mean")

        error, passed = check_gradient(
            loss_fn, grad_fn, logits, tolerance=1e-7, verbose=False
        )
        print(f"gradcheck (mean) relative error = {error:.3e}")
        assert passed, f"gradcheck failed: relative error {error:.3e} >= 1e-7"

    def test_gradcheck_passes_sum_reduction(self, capsys):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(3, 4))
        y = _one_hot(rng.integers(0, 4, size=3), 4)

        def loss_fn(z):
            return cross_entropy_from_logits(z, y, reduction="sum")

        def grad_fn(z):
            return cross_entropy_grad_from_logits(z, y, reduction="sum")

        error, passed = check_gradient(
            loss_fn, grad_fn, logits, tolerance=1e-7, verbose=False
        )
        print(f"gradcheck (sum)  relative error = {error:.3e}")
        assert passed, f"gradcheck failed: relative error {error:.3e} >= 1e-7"

    def test_gradcheck_1d_input(self, capsys):
        rng = np.random.default_rng(8)
        logits = rng.normal(size=6)
        y = np.zeros(6)
        y[rng.integers(0, 6)] = 1.0

        def loss_fn(z):
            return cross_entropy_from_logits(z, y, reduction="mean")

        def grad_fn(z):
            return cross_entropy_grad_from_logits(z, y, reduction="mean")

        error, passed = check_gradient(
            loss_fn, grad_fn, logits, tolerance=1e-7, verbose=False
        )
        print(f"gradcheck (1-D)  relative error = {error:.3e}")
        assert passed

    def test_gradient_no_batch_dim_preserved(self):
        # 1-D in should give 1-D out.
        logits = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0])
        grad = cross_entropy_grad_from_logits(logits, y, reduction="mean")
        assert grad.shape == (3,)
