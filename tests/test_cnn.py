"""Tests for the Phase 8 CNN — small network gradient check + shape smoke tests."""

import numpy as np
import pytest

from src.cnn import CNN
from src.gradient_check import numerical_gradient, relative_error
from src.loss import cross_entropy_from_logits


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


class TestForward:
    def test_output_shape(self):
        net = CNN(seed=0)
        X = np.zeros((3, 1, 28, 28))
        probs = net.forward(X)
        assert probs.shape == (3, 10)

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        net = CNN(seed=0)
        X = rng.normal(size=(4, 1, 28, 28))
        probs = net.forward(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), rtol=1e-12)

    def test_rejects_wrong_input_shape(self):
        net = CNN(seed=0)
        with pytest.raises(ValueError):
            net.forward(np.zeros((3, 3, 28, 28)))


class TestBackward:
    def test_populates_parameter_gradients(self):
        rng = np.random.default_rng(0)
        net = CNN(seed=0)
        X = rng.normal(size=(2, 1, 28, 28))
        y = _one_hot(np.array([3, 7]), 10)
        net.loss_and_grads(X, y)
        for layer in net.trainable_layers:
            assert layer.dW is not None and layer.dW.shape == layer.W.shape
            assert layer.db is not None and layer.db.shape == layer.b.shape
            assert np.all(np.isfinite(layer.dW))
            assert np.all(np.isfinite(layer.db))

    def test_params_and_grads_ordering(self):
        net = CNN(seed=0)
        X = np.zeros((1, 1, 28, 28))
        y = _one_hot(np.array([0]), 10)
        net.loss_and_grads(X, y)
        pg = net.params_and_grads()
        assert len(pg) == 6
        # W then b, layer by layer.
        for i, layer in enumerate(net.trainable_layers):
            assert pg[2 * i][0] is layer.W
            assert pg[2 * i + 1][0] is layer.b

    def test_backward_before_forward_raises(self):
        with pytest.raises(RuntimeError):
            CNN(seed=0).backward(_one_hot(np.array([0]), 10))


class TestEndToEndGradientCheck:
    """Every trainable parameter tensor's analytic gradient must match the
    numerical gradient below the Phase 4 tolerance of 1e-5.

    Uses N=2 to keep finite differences cheap while still exercising the
    batch reduction.
    """

    def test_all_parameters_below_1e5(self, capsys):
        rng = np.random.default_rng(0)
        net = CNN(seed=0)
        X = rng.normal(size=(2, 1, 28, 28)) * 0.1
        y = _one_hot(np.array([0, 5]), 10)

        # Analytic pass.
        net.forward(X)
        analytic_loss = cross_entropy_from_logits(
            net.logits_cache, y, reduction="mean"
        )
        assert np.isfinite(analytic_loss)
        net.backward(y)

        # For each trainable tensor, replace values in place and recompute loss.
        param_specs = [
            ("conv1.W", net.conv1.W, net.conv1.dW),
            ("conv1.b", net.conv1.b, net.conv1.db),
            ("conv2.W", net.conv2.W, net.conv2.dW),
            ("conv2.b", net.conv2.b, net.conv2.db),
            ("head.W",  net.head.W,  net.head.dW),
            ("head.b",  net.head.b,  net.head.db),
        ]

        max_err = 0.0
        for name, param, grad in param_specs:
            original = param.copy()

            def loss_at(flat):
                param.flat[:] = flat
                net.forward(X)
                return cross_entropy_from_logits(
                    net.logits_cache, y, reduction="mean"
                )

            num_grad = numerical_gradient(loss_at, original.flatten())
            param.flat[:] = original.flatten()

            err = relative_error(grad.flatten(), num_grad)
            print(f"{name:<9} rel_err = {err:.3e}")
            assert err < 1e-5, f"{name} failed: rel_err {err:.3e} >= 1e-5"
            max_err = max(max_err, err)

        print(f"max CNN rel_err across params = {max_err:.3e}")
