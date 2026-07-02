"""Tests for the Dense layer: He initialization, forward pass, caching."""

import numpy as np
from src.layers import Dense


class TestDenseInit:
    """Tests for parameter initialization."""

    def test_parameter_shapes(self):
        layer = Dense(784, 128, seed=0)
        assert layer.W.shape == (784, 128)
        assert layer.b.shape == (128,)

    def test_biases_zero(self):
        layer = Dense(64, 10, seed=0)
        np.testing.assert_array_equal(layer.b, np.zeros(10))

    def test_seed_reproducible(self):
        a = Dense(50, 30, seed=123)
        b = Dense(50, 30, seed=123)
        np.testing.assert_array_equal(a.W, b.W)

    def test_different_seeds_differ(self):
        a = Dense(50, 30, seed=1)
        b = Dense(50, 30, seed=2)
        assert not np.array_equal(a.W, b.W)

    def test_he_init_std(self):
        # He init: std = sqrt(2 / fan_in). Check empirically on a large layer.
        fan_in = 1000
        layer = Dense(fan_in, 5000, seed=7)
        expected_std = np.sqrt(2.0 / fan_in)
        np.testing.assert_allclose(layer.W.std(), expected_std, rtol=0.05)
        np.testing.assert_allclose(layer.W.mean(), 0.0, atol=0.01)


class TestDenseForward:
    """Tests for the forward pass and caching."""

    def test_output_shape(self):
        layer = Dense(784, 128, seed=0)
        X = np.zeros((32, 784))
        z = layer.forward(X)
        assert z.shape == (32, 128)

    def test_affine_correctness(self):
        layer = Dense(3, 2, seed=0)
        layer.W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        layer.b = np.array([0.5, -0.5])
        X = np.array([[1.0, 2.0, 3.0]])
        expected = X @ layer.W + layer.b
        np.testing.assert_allclose(layer.forward(X), expected)

    def test_caches_input_and_z(self):
        layer = Dense(4, 3, seed=0)
        X = np.ones((2, 4))
        z = layer.forward(X)
        np.testing.assert_array_equal(layer.input_cache, X)
        np.testing.assert_array_equal(layer.z_cache, z)

    def test_caches_none_before_forward(self):
        layer = Dense(4, 3, seed=0)
        assert layer.input_cache is None
        assert layer.z_cache is None

    def test_wrong_input_dim_raises(self):
        layer = Dense(4, 3, seed=0)
        import pytest

        with pytest.raises(ValueError):
            layer.forward(np.zeros((2, 5)))
