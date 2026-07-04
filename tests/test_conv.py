"""Tests for the Phase 8 Conv2D layer (im2col-vectorized 2D convolution)."""

import numpy as np
import pytest

from src.conv import Conv2D, _col2im, _im2col
from src.gradient_check import numerical_gradient, relative_error
from src.loss import cross_entropy_from_logits, cross_entropy_grad_from_logits


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


class TestOutputShapes:
    """Output spatial dims obey (H + 2p - k)/s + 1."""

    @pytest.mark.parametrize(
        "N,C_in,C_out,H,W,k,s,p,H_out,W_out",
        [
            (2, 1, 4, 8, 8, 3, 1, 0, 6, 6),
            (2, 1, 4, 8, 8, 3, 1, 1, 8, 8),
            (2, 3, 5, 10, 10, 3, 2, 0, 4, 4),
            (2, 3, 5, 10, 10, 3, 2, 1, 5, 5),
            (1, 2, 2, 5, 5, 5, 1, 0, 1, 1),
        ],
    )
    def test_forward_shape(self, N, C_in, C_out, H, W, k, s, p, H_out, W_out):
        layer = Conv2D(C_in, C_out, k, stride=s, padding=p, seed=0)
        X = np.zeros((N, C_in, H, W))
        out = layer.forward(X)
        assert out.shape == (N, C_out, H_out, W_out)


class TestForwardCorrectness:
    def test_matches_manual_convolution(self):
        """Cross-check the im2col forward against a hand-written naïve loop."""
        rng = np.random.default_rng(0)
        N, C_in, C_out, H, W, k = 2, 3, 4, 6, 6, 3
        X = rng.normal(size=(N, C_in, H, W))

        layer = Conv2D(C_in, C_out, k, stride=1, padding=0, seed=1)
        vectorized = layer.forward(X)

        # Naïve reference.
        H_out = H - k + 1
        W_out = W - k + 1
        naive = np.zeros((N, C_out, H_out, W_out))
        for n in range(N):
            for co in range(C_out):
                for y in range(H_out):
                    for x in range(W_out):
                        acc = layer.b[co]
                        for ci in range(C_in):
                            for ky in range(k):
                                for kx in range(k):
                                    acc += (
                                        X[n, ci, y + ky, x + kx]
                                        * layer.W[co, ci, ky, kx]
                                    )
                        naive[n, co, y, x] = acc

        np.testing.assert_allclose(vectorized, naive, rtol=1e-12)

    def test_padding_shifts_output(self):
        # Padding 1 with kernel 3 preserves spatial dims.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1, 1, 4, 4))
        layer = Conv2D(1, 2, kernel_size=3, padding=1, seed=2)
        out = layer.forward(X)
        assert out.shape == (1, 2, 4, 4)


class TestIm2ColRoundTrip:
    def test_col2im_recovers_ones_at_overlaps(self):
        """A ``cols`` of all-ones col2im's to the number of windows each
        input position participates in — a sanity check on window counting.
        """
        X_shape = (1, 1, 4, 4)
        kH = kW = 3
        stride = 1
        pad = 0
        H_out = 4 - kH + 1
        W_out = 4 - kW + 1
        n_rows = 1 * H_out * W_out
        cols = np.ones((n_rows, 1 * kH * kW), dtype=np.float64)
        dX = _col2im(cols, X_shape, kH, kW, stride, pad, H_out, W_out)
        # The center of a 4x4 with 3x3 kernel & stride 1 is covered 4 times.
        assert dX[0, 0, 1, 1] == 4.0
        # Corner is covered once.
        assert dX[0, 0, 0, 0] == 1.0

    def test_im2col_row_count(self):
        X = np.zeros((2, 3, 5, 5))
        cols, (H_out, W_out) = _im2col(X, 3, 3, stride=1, pad=0)
        assert cols.shape == (2 * H_out * W_out, 3 * 3 * 3)


class TestConvGradientCheck:
    """Analytic gradients pass the Phase 1 finite-difference check.

    Uses a tiny problem — small spatial dims, small channels, small batch —
    so numerical differentiation is affordable.
    """

    def _make_problem(self, seed):
        rng = np.random.default_rng(seed)
        N, C_in, C_out, H, W, k = 2, 2, 3, 4, 4, 3
        X = rng.normal(size=(N, C_in, H, W))
        # Use a fake "loss" of sum-of-square outputs, since the layer isn't
        # attached to a classifier here — this is enough to check the
        # backward math on W, b, and X independently of downstream code.
        return X, N, C_in, C_out, H, W, k

    def test_gradient_wrt_input(self):
        X, N, C_in, C_out, H, W, k = self._make_problem(0)
        layer = Conv2D(C_in, C_out, k, stride=1, padding=0, seed=1)

        def loss_at(x_flat):
            out = layer.forward(x_flat.reshape(X.shape))
            return 0.5 * float(np.sum(out * out))

        out = layer.forward(X)
        dout = out.copy()  # gradient of 0.5 * sum(out^2) w.r.t. out is out
        analytic_dX = layer.backward(dout)

        num_dX = numerical_gradient(loss_at, X.flatten())
        err = relative_error(analytic_dX.flatten(), num_dX)
        assert err < 1e-6, f"conv dX rel_err = {err:.3e}"

    def test_gradient_wrt_weights(self):
        X, N, C_in, C_out, H, W, k = self._make_problem(1)
        layer = Conv2D(C_in, C_out, k, stride=1, padding=0, seed=2)
        W_original = layer.W.copy()

        def loss_at(W_flat):
            layer.W = W_flat.reshape(W_original.shape)
            out = layer.forward(X)
            return 0.5 * float(np.sum(out * out))

        out = layer.forward(X)
        dout = out.copy()
        layer.backward(dout)
        analytic_dW = layer.dW.copy()

        num_dW = numerical_gradient(loss_at, W_original.flatten())
        layer.W = W_original  # restore
        err = relative_error(analytic_dW.flatten(), num_dW)
        assert err < 1e-6, f"conv dW rel_err = {err:.3e}"

    def test_gradient_wrt_bias(self):
        X, N, C_in, C_out, H, W, k = self._make_problem(2)
        layer = Conv2D(C_in, C_out, k, stride=1, padding=0, seed=3)
        b_original = layer.b.copy()

        def loss_at(b_flat):
            layer.b = b_flat
            out = layer.forward(X)
            return 0.5 * float(np.sum(out * out))

        out = layer.forward(X)
        dout = out.copy()
        layer.backward(dout)
        analytic_db = layer.db.copy()

        num_db = numerical_gradient(loss_at, b_original)
        layer.b = b_original
        err = relative_error(analytic_db, num_db)
        assert err < 1e-6, f"conv db rel_err = {err:.3e}"

    def test_gradient_check_with_padding_and_ce_head(self, capsys):
        """Attach conv to a linear head + softmax + cross-entropy and check
        every conv parameter against the Phase 1 utility at tolerance 1e-5
        (the deeper composition tolerance)."""
        rng = np.random.default_rng(4)
        N, C_in, C_out, H, W, k = 2, 1, 2, 4, 4, 3
        X = rng.normal(size=(N, C_in, H, W))
        # Head projects [C_out * H_out * W_out] -> num_classes.
        K = 3
        H_out = H + 2 * 1 - k + 1  # padding=1
        W_out = W + 2 * 1 - k + 1
        head_W = rng.normal(size=(C_out * H_out * W_out, K)) * 0.1
        head_b = np.zeros(K)
        y = _one_hot(rng.integers(0, K, size=N), K)

        conv = Conv2D(C_in, C_out, k, stride=1, padding=1, seed=5)

        def forward_and_loss():
            out = conv.forward(X)
            flat = out.reshape(N, -1)
            logits = flat @ head_W + head_b
            return cross_entropy_from_logits(logits, y, reduction="mean"), logits, flat

        # Analytic pass end-to-end.
        loss, logits, flat = forward_and_loss()
        dlogits = cross_entropy_grad_from_logits(logits, y, reduction="mean")
        dflat = dlogits @ head_W.T
        dout = dflat.reshape(N, C_out, H_out, W_out)
        conv.backward(dout)

        # Numerical grad w.r.t. W.
        original_W = conv.W.copy()

        def loss_of_W(W_flat):
            conv.W = W_flat.reshape(original_W.shape)
            L, _, _ = forward_and_loss()
            return L

        num_dW = numerical_gradient(loss_of_W, original_W.flatten())
        conv.W = original_W
        err_W = relative_error(conv.dW.flatten(), num_dW)
        print(f"conv+head dW rel_err = {err_W:.3e}")
        assert err_W < 1e-5

        # Numerical grad w.r.t. b.
        original_b = conv.b.copy()

        def loss_of_b(b_flat):
            conv.b = b_flat
            L, _, _ = forward_and_loss()
            return L

        num_db = numerical_gradient(loss_of_b, original_b.flatten())
        conv.b = original_b
        err_b = relative_error(conv.db.flatten(), num_db)
        print(f"conv+head db rel_err = {err_b:.3e}")
        assert err_b < 1e-5


class TestGuards:
    def test_forward_rejects_bad_channel_count(self):
        layer = Conv2D(3, 4, 3, seed=0)
        with pytest.raises(ValueError):
            layer.forward(np.zeros((1, 5, 6, 6)))

    def test_backward_before_forward_raises(self):
        layer = Conv2D(1, 1, 3, seed=0)
        with pytest.raises(RuntimeError):
            layer.backward(np.zeros((1, 1, 2, 2)))
