"""2D convolutional layer (Phase 8, stretch).

Implements a NumPy-only ``Conv2D`` layer with an im2col-vectorized
forward and backward pass. The naïve nested-loop derivation is:

    for n, c_out, y_out, x_out:
        acc = b[c_out]
        for c_in, ky, kx:
            acc += X[n, c_in, y_out*s + ky, x_out*s + kx] * W[c_out, c_in, ky, kx]
        out[n, c_out, y_out, x_out] = acc

The vectorized form flattens each receptive field into a row of a
matrix (im2col), which turns convolution into a single ``matmul`` and
reuses NumPy's BLAS-backed GEMM. col2im scatters gradients back into
the input tensor at the same window positions, correctly accumulating
in regions of overlap.

Weight layout: ``W`` has shape ``[C_out, C_in, kH, kW]`` (matching the
naïve loop above); ``b`` has shape ``[C_out]``. Weights use He
initialization on the flattened receptive-field dimension.
"""

from __future__ import annotations

import numpy as np


def _im2col(X: np.ndarray, kH: int, kW: int, stride: int, pad: int):
    """Turn ``[N, C, H, W]`` input into a matrix of stretched receptive fields.

    Returns ``(cols, (H_out, W_out))`` where ``cols`` has shape
    ``[N * H_out * W_out, C * kH * kW]``. Each row is one flattened
    ``(C, kH, kW)`` window; rows are ordered by ``n, y_out, x_out``.
    """
    N, C, H, W = X.shape
    if pad > 0:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1

    windows = np.lib.stride_tricks.sliding_window_view(X, (kH, kW), axis=(2, 3))
    # shape: [N, C, H_padded - kH + 1, W_padded - kW + 1, kH, kW]
    windows = windows[:, :, ::stride, ::stride, :, :]  # apply stride
    # -> [N, C, H_out, W_out, kH, kW]

    cols = (
        windows.transpose(0, 2, 3, 1, 4, 5)  # [N, H_out, W_out, C, kH, kW]
        .reshape(N * H_out * W_out, C * kH * kW)
    )
    return cols, (H_out, W_out)


def _col2im(
    cols: np.ndarray,
    X_shape: tuple[int, int, int, int],
    kH: int,
    kW: int,
    stride: int,
    pad: int,
    H_out: int,
    W_out: int,
) -> np.ndarray:
    """Inverse of :func:`_im2col` — sums gradients back into an ``X``-shaped tensor.

    ``cols`` has shape ``[N * H_out * W_out, C * kH * kW]``. Overlapping
    receptive fields sum their contributions in the returned dX.
    """
    N, C, H, W = X_shape
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad

    cols_reshaped = (
        cols.reshape(N, H_out, W_out, C, kH, kW)
        .transpose(0, 3, 1, 2, 4, 5)  # [N, C, H_out, W_out, kH, kW]
    )

    dX_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    for i in range(kH):
        i_end = i + stride * H_out
        for j in range(kW):
            j_end = j + stride * W_out
            dX_padded[:, :, i:i_end:stride, j:j_end:stride] += (
                cols_reshaped[:, :, :, :, i, j]
            )

    if pad > 0:
        return dX_padded[:, :, pad : pad + H, pad : pad + W]
    return dX_padded


class Conv2D:
    """2D convolution layer computing ``out = conv(X, W) + b``.

    Args:
        in_channels: number of input channels ``C_in``.
        out_channels: number of output channels ``C_out``.
        kernel_size: side length of the square kernel.
        stride: convolution stride (same for both spatial dims). Default: 1.
        padding: zero-padding applied symmetrically to both spatial dims.
            Default: 0.
        seed: optional int for reproducible weight initialization.

    Attributes:
        W: kernel tensor, shape ``[C_out, C_in, kH, kW]``.
        b: bias vector, shape ``[C_out]``.
        input_shape: cached shape of the last input tensor to forward().
        cols_cache: cached im2col of the last input, used by backward().
        dW, db: parameter gradients populated by backward().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        seed: int | None = None,
    ):
        self.C_in = in_channels
        self.C_out = out_channels
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(
            0.0, std, size=(out_channels, in_channels, kernel_size, kernel_size)
        ).astype(np.float64)
        self.b = np.zeros(out_channels, dtype=np.float64)

        self.input_shape = None
        self.cols_cache = None
        self.out_hw_cache = None
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Convolve ``X`` with the layer's kernel.

        Args:
            X: input tensor of shape ``[N, C_in, H, W]``.

        Returns:
            Output tensor of shape ``[N, C_out, H_out, W_out]``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4 or X.shape[1] != self.C_in:
            raise ValueError(
                f"Expected input of shape [N, {self.C_in}, H, W]; got {X.shape}"
            )

        self.input_shape = X.shape
        cols, (H_out, W_out) = _im2col(
            X, self.kH, self.kW, self.stride, self.padding
        )
        self.cols_cache = cols
        self.out_hw_cache = (H_out, W_out)

        # Weight matrix: [C_out, C_in * kH * kW]. cols @ W_mat.T -> per-window logits.
        W_mat = self.W.reshape(self.C_out, -1)
        out = cols @ W_mat.T + self.b  # broadcasting adds bias per C_out
        N = X.shape[0]
        return out.reshape(N, H_out, W_out, self.C_out).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backpropagate through the convolution.

        Args:
            dout: gradient w.r.t. the output, shape ``[N, C_out, H_out, W_out]``.

        Returns:
            ``dX``: gradient w.r.t. the input, shape matching the last forward
            input.
        """
        if self.cols_cache is None:
            raise RuntimeError("backward() called before forward()")

        N, C_out, H_out, W_out = dout.shape
        if (H_out, W_out) != self.out_hw_cache or C_out != self.C_out:
            raise ValueError(
                f"dout shape {dout.shape} inconsistent with cached forward "
                f"output {(N, self.C_out) + self.out_hw_cache}"
            )

        # Flatten dout into rows matching the cols layout: [N*H_out*W_out, C_out].
        dout_mat = dout.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)

        # Bias gradient: sum over N * H_out * W_out.
        self.db = dout_mat.sum(axis=0)

        # Weight gradient: (dout_mat).T @ cols -> [C_out, C_in * kH * kW].
        dW_mat = dout_mat.T @ self.cols_cache
        self.dW = dW_mat.reshape(self.W.shape)

        # Input gradient: (dout_mat @ W_mat) -> [N*H_out*W_out, C_in*kH*kW]
        # then col2im back to input shape.
        W_mat = self.W.reshape(self.C_out, -1)
        dcols = dout_mat @ W_mat
        dX = _col2im(
            dcols,
            self.input_shape,
            self.kH,
            self.kW,
            self.stride,
            self.padding,
            H_out,
            W_out,
        )
        return dX
