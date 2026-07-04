"""Max-pooling and flattening layers (Phase 8, stretch).

Both layers are parameter-free — they have no ``W`` / ``b`` and expose
no ``dW`` / ``db``. They still expose ``forward`` and ``backward`` so
they compose with :class:`src.conv.Conv2D` and :class:`src.layers.Dense`
inside a CNN.
"""

from __future__ import annotations

import numpy as np


class MaxPool2D:
    """Non-overlapping 2D max pooling (when ``stride == kernel_size``).

    Args:
        kernel_size: side length of the square pooling window.
        stride: pool stride. Defaults to ``kernel_size`` (non-overlapping).
    """

    def __init__(self, kernel_size: int, stride: int | None = None):
        self.kH = kernel_size
        self.kW = kernel_size
        self.stride = kernel_size if stride is None else stride

        self.input_shape = None
        self.argmax_cache = None  # flat index within each window
        self.out_hw_cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Take the max over each non-overlapping window.

        Args:
            X: input of shape ``[N, C, H, W]``.

        Returns:
            Output of shape ``[N, C, H_out, W_out]``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4:
            raise ValueError(f"Expected 4-D input; got {X.shape}")

        N, C, H, W = X.shape
        s = self.stride
        kH, kW = self.kH, self.kW
        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        windows = np.lib.stride_tricks.sliding_window_view(X, (kH, kW), axis=(2, 3))
        # [N, C, H - kH + 1, W - kW + 1, kH, kW]
        windows = windows[:, :, ::s, ::s, :, :]  # [N, C, H_out, W_out, kH, kW]

        # Flatten each window to get a single argmax per window (ties broken
        # by argmax's default: lowest index wins).
        flat = windows.reshape(N, C, H_out, W_out, kH * kW)
        argmax = flat.argmax(axis=-1)  # [N, C, H_out, W_out]
        out = np.take_along_axis(flat, argmax[..., None], axis=-1).squeeze(-1)

        self.input_shape = X.shape
        self.argmax_cache = argmax
        self.out_hw_cache = (H_out, W_out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Route each output gradient back to the argmax input position.

        Args:
            dout: gradient w.r.t. the output, shape ``[N, C, H_out, W_out]``.

        Returns:
            Gradient w.r.t. the input, shape matching the last forward input.
        """
        if self.argmax_cache is None:
            raise RuntimeError("backward() called before forward()")

        N, C, H, W = self.input_shape
        H_out, W_out = self.out_hw_cache
        s = self.stride
        kH, kW = self.kH, self.kW

        argmax_h, argmax_w = np.unravel_index(self.argmax_cache, (kH, kW))
        # argmax_h, argmax_w: [N, C, H_out, W_out]

        h_out_idx = np.arange(H_out).reshape(1, 1, H_out, 1)
        w_out_idx = np.arange(W_out).reshape(1, 1, 1, W_out)
        abs_h = h_out_idx * s + argmax_h  # broadcasts to [N, C, H_out, W_out]
        abs_w = w_out_idx * s + argmax_w

        n_idx = np.broadcast_to(
            np.arange(N).reshape(N, 1, 1, 1), (N, C, H_out, W_out)
        )
        c_idx = np.broadcast_to(
            np.arange(C).reshape(1, C, 1, 1), (N, C, H_out, W_out)
        )

        dX = np.zeros(self.input_shape, dtype=np.float64)
        np.add.at(dX, (n_idx, c_idx, abs_h, abs_w), dout)
        return dX


class Flatten:
    """Flatten each sample's non-batch dims into a single vector.

    Input ``[N, ...]`` -> output ``[N, prod(...)]``. Records the original
    shape so backward() can restore it.
    """

    def __init__(self):
        self.input_shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise RuntimeError("backward() called before forward()")
        return dout.reshape(self.input_shape)
