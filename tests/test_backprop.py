"""Tests for the Phase 4 backward pass.

Coverage:

- ``relu_grad`` correctness (known values, zero convention, shape).
- ``Dense.backward`` correctness — output shapes and end-to-end analytical
  vs numerical gradient on a standalone Dense + softmax + cross-entropy
  micro-graph.
- Full-network gradient check on a small MLP with the same architecture
  as ``Network`` but tiny dims, so finite-difference gradient checking
  finishes in fractions of a second. Every weight and bias tensor is
  compared to its numerical gradient with a relative-error tolerance of
  ``1e-5`` (per the Phase 4 exit criterion). Per-parameter relative
  errors are printed.
- Sanity check on the real ``Network(seed=0)`` at MNIST dimensions with
  a tiny batch: forward → backward runs, gradients have the correct
  shapes, and no NaN / Inf appears.
"""

import numpy as np
import pytest

from src.activations import relu, relu_grad, softmax
from src.gradient_check import numerical_gradient, relative_error
from src.layers import Dense
from src.loss import cross_entropy_from_logits
from src.network import Network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


class _MiniNet:
    """A tiny MLP with the same shape (Dense-ReLU-Dense-ReLU-Dense-Softmax)
    as ``src.network.Network`` but with configurable dims so finite-difference
    gradient checking is affordable.

    We deliberately reuse the production ``Dense`` layer and activation
    derivative so the test exercises the same code paths as the real network.
    """

    def __init__(self, dims, seed=0):
        assert len(dims) == 4, "MiniNet expects [in, h1, h2, out]"
        self.layers = [
            Dense(dims[0], dims[1], seed=seed),
            Dense(dims[1], dims[2], seed=seed + 1),
            Dense(dims[2], dims[3], seed=seed + 2),
        ]
        self.logits_cache = None

    def forward(self, X):
        h1 = relu(self.layers[0].forward(X))
        h2 = relu(self.layers[1].forward(h1))
        logits = self.layers[2].forward(h2)
        self.logits_cache = logits
        return softmax(logits)

    def loss(self, X, y):
        self.forward(X)
        return cross_entropy_from_logits(self.logits_cache, y, reduction="mean")

    def backward(self, y):
        from src.loss import cross_entropy_grad_from_logits

        dlogits = cross_entropy_grad_from_logits(
            self.logits_cache, y, reduction="mean"
        )
        dh2 = self.layers[2].backward(dlogits)
        dz2 = dh2 * relu_grad(self.layers[1].z_cache)
        dh1 = self.layers[1].backward(dz2)
        dz1 = dh1 * relu_grad(self.layers[0].z_cache)
        self.layers[0].backward(dz1)


def _param_refs(net):
    """Yield ``(name, param_array, gradient_attr_name, layer)`` for every
    trainable parameter tensor in a MiniNet or Network."""
    for i, layer in enumerate(net.layers):
        yield f"layers[{i}].W", layer.W, "dW", layer
        yield f"layers[{i}].b", layer.b, "db", layer


# ---------------------------------------------------------------------------
# relu_grad
# ---------------------------------------------------------------------------


class TestReLUGrad:
    def test_known_values(self):
        z = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(relu_grad(z), expected)

    def test_zero_convention(self):
        # Subgradient at 0 chosen as 0 (matches most frameworks).
        assert relu_grad(np.array([0.0]))[0] == 0.0

    def test_shape_preserved(self):
        z = np.zeros((3, 5))
        assert relu_grad(z).shape == (3, 5)

    def test_binary_output(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(50, 50))
        out = relu_grad(z)
        # All entries are exactly 0 or 1.
        assert np.all((out == 0.0) | (out == 1.0))


# ---------------------------------------------------------------------------
# Dense.backward — standalone
# ---------------------------------------------------------------------------


class TestDenseBackward:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        layer = Dense(6, 4, seed=0)
        X = rng.normal(size=(3, 6))
        layer.forward(X)
        dz = rng.normal(size=(3, 4))
        dX = layer.backward(dz)
        assert layer.dW.shape == layer.W.shape
        assert layer.db.shape == layer.b.shape
        assert dX.shape == X.shape

    def test_backward_before_forward_raises(self):
        layer = Dense(3, 2, seed=0)
        with pytest.raises(RuntimeError):
            layer.backward(np.zeros((1, 2)))

    def test_dz_shape_mismatch_raises(self):
        layer = Dense(3, 2, seed=0)
        layer.forward(np.zeros((4, 3)))
        with pytest.raises(ValueError):
            layer.backward(np.zeros((4, 3)))  # wrong second dim

    def test_gradients_match_numerical_single_layer(self, capsys):
        """One Dense layer -> softmax -> cross-entropy, end-to-end gradcheck.

        This isolates ``Dense.backward``: the only thing between logits and
        loss is the fused softmax+CE derivative, which is well-tested in
        Phase 3, so any error here is attributable to Dense.
        """
        rng = np.random.default_rng(1)
        N, D, K = 4, 5, 3
        X = rng.normal(size=(N, D))
        y = _one_hot(rng.integers(0, K, size=N), K)

        layer = Dense(D, K, seed=42)

        def loss_of_params():
            logits = layer.forward(X)
            return cross_entropy_from_logits(logits, y, reduction="mean")

        # Analytic gradients via one forward+backward.
        from src.loss import cross_entropy_grad_from_logits

        logits = layer.forward(X)
        dlogits = cross_entropy_grad_from_logits(logits, y, reduction="mean")
        layer.backward(dlogits)

        # Numerical gradients for W.
        original_W = layer.W.copy()

        def loss_at_W(W_flat):
            layer.W = W_flat.reshape(original_W.shape)
            return loss_of_params()

        num_dW = numerical_gradient(loss_at_W, original_W.flatten())
        layer.W = original_W  # restore
        err_W = relative_error(layer.dW.flatten(), num_dW)
        print(f"Dense.backward standalone W rel_err = {err_W:.3e}")
        assert err_W < 1e-7

        # Numerical gradients for b.
        original_b = layer.b.copy()

        def loss_at_b(b_flat):
            layer.b = b_flat
            return loss_of_params()

        num_db = numerical_gradient(loss_at_b, original_b.flatten())
        layer.b = original_b
        err_b = relative_error(layer.db.flatten(), num_db)
        print(f"Dense.backward standalone b rel_err = {err_b:.3e}")
        assert err_b < 1e-7


# ---------------------------------------------------------------------------
# End-to-end network gradient check (exit criterion)
# ---------------------------------------------------------------------------


class TestFullNetworkGradients:
    """Compare analytic vs numerical gradients for every parameter tensor
    in a small MLP with the same topology as ``Network``.
    """

    def _check_all(self, net, X, y, tol, label):
        # Analytical pass: populates layer.dW / layer.db.
        net.forward(X)
        analytic_loss = cross_entropy_from_logits(
            net.logits_cache, y, reduction="mean"
        )
        assert np.isfinite(analytic_loss)
        net.backward(y)

        # Numerical pass: perturb each parameter tensor element-wise.
        max_err = 0.0
        for name, param, grad_attr, layer in _param_refs(net):
            original = param.copy()
            analytic_grad = getattr(layer, grad_attr)
            assert analytic_grad.shape == param.shape

            def loss_at(param_flat):
                param.flat[:] = param_flat
                net.forward(X)
                return cross_entropy_from_logits(
                    net.logits_cache, y, reduction="mean"
                )

            num_grad = numerical_gradient(loss_at, original.flatten())
            param.flat[:] = original.flatten()  # restore

            err = relative_error(analytic_grad.flatten(), num_grad)
            print(f"[{label}] {name:<14} rel_err = {err:.3e}")
            assert err < tol, (
                f"{label} {name} failed: rel_err {err:.3e} >= {tol:.0e}"
            )
            max_err = max(max_err, err)

        print(f"[{label}] max rel_err across params = {max_err:.3e}")

    def test_mini_network_all_params_below_1e5(self, capsys):
        """Exit criterion — every parameter's analytic grad matches
        numerical to relative error < 1e-5 on a small end-to-end network.
        """
        rng = np.random.default_rng(2)
        dims = [6, 5, 4, 3]  # in, h1, h2, out
        net = _MiniNet(dims, seed=7)
        X = rng.normal(size=(4, dims[0]))
        y = _one_hot(rng.integers(0, dims[-1], size=4), dims[-1])
        self._check_all(net, X, y, tol=1e-5, label="mini")

    def test_batch_of_one(self, capsys):
        """The mean-reduction gradient must still be correct when N=1."""
        rng = np.random.default_rng(3)
        dims = [4, 5, 3, 3]
        net = _MiniNet(dims, seed=11)
        X = rng.normal(size=(1, dims[0]))
        y = _one_hot(np.array([1]), dims[-1])
        self._check_all(net, X, y, tol=1e-5, label="N=1")


# ---------------------------------------------------------------------------
# Real Network sanity — MNIST dims, tiny batch, no numerical gradient
# (finite differences on ~110k params is not test-time-affordable).
# ---------------------------------------------------------------------------


class TestRealNetworkBackward:
    def test_gradients_shapes_and_finite(self):
        net = Network(seed=0)
        rng = np.random.default_rng(4)
        X = rng.normal(size=(3, 784))
        y = _one_hot(rng.integers(0, 10, size=3), 10)

        loss = net.loss_and_grads(X, y)
        assert np.isfinite(loss)

        for layer in net.layers:
            assert layer.dW.shape == layer.W.shape
            assert layer.db.shape == layer.b.shape
            assert np.all(np.isfinite(layer.dW))
            assert np.all(np.isfinite(layer.db))

    def test_backward_before_forward_raises(self):
        net = Network(seed=0)
        y = _one_hot(np.array([0]), 10)
        with pytest.raises(RuntimeError):
            net.backward(y)

    def test_zero_grad_when_label_matches_argmax_confidently(self):
        """When the network is (artificially) perfectly confident and
        correct, the softmax+CE gradient should collapse to (near-)zero,
        and therefore every parameter gradient should be near-zero too.
        """
        net = Network(seed=0)
        # Zero all weights and biases except set output-layer bias to a
        # huge value on class 0 -> softmax is essentially [1, 0, ..., 0].
        for layer in net.layers:
            layer.W[:] = 0.0
            layer.b[:] = 0.0
        net.layers[2].b[0] = 1e3

        X = np.zeros((2, 784))
        y = _one_hot(np.array([0, 0]), 10)
        net.loss_and_grads(X, y)

        for layer in net.layers:
            assert np.max(np.abs(layer.dW)) < 1e-6
            assert np.max(np.abs(layer.db)) < 1e-6
