"""Tests for activation functions (ReLU, sigmoid, numerically stable softmax)."""

import numpy as np
from src.activations import relu, sigmoid, softmax


class TestReLU:
    """Tests for the ReLU activation."""

    def test_known_values(self):
        z = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
        expected = np.array([0.0, 0.0, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(relu(z), expected)

    def test_nonnegative_output(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(20, 20)) * 100
        assert np.all(relu(z) >= 0.0)

    def test_shape_preserved(self):
        z = np.zeros((4, 7))
        assert relu(z).shape == (4, 7)


class TestSigmoid:
    """Tests for the logistic sigmoid activation."""

    def test_zero_is_half(self):
        np.testing.assert_allclose(sigmoid(np.array([0.0])), np.array([0.5]))

    def test_known_values(self):
        z = np.array([-1.0, 0.0, 1.0])
        expected = 1.0 / (1.0 + np.exp(-z))
        np.testing.assert_allclose(sigmoid(z), expected)

    def test_range_bounded(self):
        # Moderate inputs stay strictly inside (0, 1).
        rng = np.random.default_rng(1)
        z = rng.normal(size=1000) * 5
        out = sigmoid(z)
        assert np.all(out > 0.0) and np.all(out < 1.0)

    def test_range_bounded_extreme(self):
        # Large-magnitude inputs saturate to the closed interval [0, 1]
        # in float64 without overflow.
        rng = np.random.default_rng(1)
        z = rng.normal(size=1000) * 50
        out = sigmoid(z)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_no_overflow_large_inputs(self):
        z = np.array([-1000.0, 1000.0])
        out = sigmoid(z)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, np.array([0.0, 1.0]), atol=1e-12)


class TestSoftmax:
    """Tests for the numerically stable row-wise softmax."""

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=(5, 10))
        out = softmax(z)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5))

    def test_entries_in_unit_interval(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(8, 10)) * 5
        out = softmax(z)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_known_values(self):
        z = np.array([[1.0, 2.0, 3.0]])
        e = np.exp(z - z.max())
        expected = e / e.sum()
        np.testing.assert_allclose(softmax(z), expected)

    def test_stability_large_inputs(self):
        # Without max-subtraction, exp(10000) overflows to inf.
        z = np.array([[10000.0, 10001.0, 10002.0]])
        out = softmax(z)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(1))

    def test_shift_invariance(self):
        z = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(softmax(z), softmax(z + 50.0))

    def test_1d_input(self):
        z = np.array([1.0, 2.0, 3.0])
        out = softmax(z)
        assert out.shape == (3,)
        np.testing.assert_allclose(out.sum(), 1.0)
