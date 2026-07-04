"""Tests for the Phase 8 MaxPool2D and Flatten layers."""

import numpy as np
import pytest

from src.gradient_check import numerical_gradient, relative_error
from src.pooling import Flatten, MaxPool2D


class TestMaxPoolForward:
    def test_shape_non_overlapping(self):
        X = np.zeros((2, 3, 8, 8))
        out = MaxPool2D(kernel_size=2).forward(X)
        assert out.shape == (2, 3, 4, 4)

    def test_shape_with_stride(self):
        X = np.zeros((1, 1, 6, 6))
        out = MaxPool2D(kernel_size=3, stride=1).forward(X)
        assert out.shape == (1, 1, 4, 4)

    def test_max_values(self):
        # Concrete 4x4 -> 2x2 non-overlapping 2x2 pool.
        X = np.array(
            [[[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]]],
            dtype=np.float64,
        )
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float64)
        out = MaxPool2D(kernel_size=2).forward(X)
        np.testing.assert_allclose(out, expected)

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError):
            MaxPool2D(2).forward(np.zeros((3, 4)))


class TestMaxPoolBackward:
    def test_gradient_routes_to_argmax(self):
        pool = MaxPool2D(kernel_size=2)
        X = np.array(
            [[[
                [0, 1, 0, 0],
                [0, 0, 0, 2],
                [3, 0, 0, 0],
                [0, 0, 0, 4],
            ]]],
            dtype=np.float64,
        )
        pool.forward(X)
        dout = np.array([[[[1.0, 1.0], [1.0, 1.0]]]])
        dX = pool.backward(dout)
        # Grad should sit at (0,1) top-left window, (1,3) top-right,
        # (2,0) bottom-left, (3,3) bottom-right — the argmax positions.
        expected = np.zeros_like(X)
        expected[0, 0, 0, 1] = 1.0
        expected[0, 0, 1, 3] = 1.0
        expected[0, 0, 2, 0] = 1.0
        expected[0, 0, 3, 3] = 1.0
        np.testing.assert_allclose(dX, expected)

    def test_backward_before_forward_raises(self):
        with pytest.raises(RuntimeError):
            MaxPool2D(2).backward(np.zeros((1, 1, 1, 1)))

    def test_gradient_check(self):
        """Analytic vs numerical gradient on a small tensor."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1, 2, 4, 4))
        pool = MaxPool2D(kernel_size=2)

        def loss_at(x_flat):
            out = pool.forward(x_flat.reshape(X.shape))
            return 0.5 * float(np.sum(out * out))

        out = pool.forward(X)
        dout = out.copy()  # d(0.5 * sum out^2)/dout = out
        analytic_dX = pool.backward(dout)

        num_dX = numerical_gradient(loss_at, X.flatten())
        err = relative_error(analytic_dX.flatten(), num_dX)
        # At tied entries the numerical grad is ambiguous, but on random
        # continuous data ties are measure-zero.
        assert err < 1e-6, f"pool dX rel_err = {err:.3e}"


class TestFlatten:
    def test_forward_shape(self):
        X = np.zeros((3, 4, 5, 6))
        out = Flatten().forward(X)
        assert out.shape == (3, 4 * 5 * 6)

    def test_backward_restores_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2, 3, 4, 5))
        f = Flatten()
        out = f.forward(X)
        dX = f.backward(out)  # identity through the flatten
        np.testing.assert_allclose(dX, X)

    def test_backward_before_forward_raises(self):
        with pytest.raises(RuntimeError):
            Flatten().backward(np.zeros((1, 1)))
