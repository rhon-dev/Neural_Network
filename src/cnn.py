"""A small MNIST CNN wired from the Phase 8 conv + pool primitives.

Architecture:

    Input [N, 1, 28, 28]
      -> Conv2D(1, 8,  kernel=3, padding=1) + ReLU  -> [N, 8, 28, 28]
      -> MaxPool2D(2)                                -> [N, 8, 14, 14]
      -> Conv2D(8, 16, kernel=3, padding=1) + ReLU   -> [N, 16, 14, 14]
      -> MaxPool2D(2)                                -> [N, 16, 7, 7]
      -> Flatten                                     -> [N, 784]
      -> Dense(784, 10)                              -> [N, 10] (logits)
      -> Softmax                                     -> [N, 10] (probs)

Cross-entropy is fused with softmax on the backward path (same trick as
the MLP): the head-layer gradient is ``(softmax(logits) - y) / N``.

The layer list is intentionally kept homogeneous — everything with
trainable parameters lives at ``self.trainable_layers``, so the training
loop iterates one canonical list to hand ``(param, grad)`` pairs to an
optimizer.
"""

from __future__ import annotations

import numpy as np

from src.activations import relu, relu_grad, softmax
from src.conv import Conv2D
from src.layers import Dense
from src.loss import cross_entropy_from_logits, cross_entropy_grad_from_logits
from src.pooling import Flatten, MaxPool2D

INPUT_SHAPE = (1, 28, 28)


class CNN:
    """Small convolutional network for MNIST classification."""

    def __init__(self, seed: int = 0):
        # Seed each parameterized layer independently but reproducibly.
        self.conv1 = Conv2D(1, 8, kernel_size=3, padding=1, seed=seed)
        self.pool1 = MaxPool2D(kernel_size=2)
        self.conv2 = Conv2D(8, 16, kernel_size=3, padding=1, seed=seed + 1)
        self.pool2 = MaxPool2D(kernel_size=2)
        self.flatten = Flatten()
        self.head = Dense(16 * 7 * 7, 10, seed=seed + 2)

        # For backward through the ReLU non-linearities, cache their inputs.
        self._z_conv1 = None
        self._z_conv2 = None

        # Populated by forward().
        self.logits_cache = None

    @property
    def trainable_layers(self):
        return [self.conv1, self.conv2, self.head]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run the forward pass; return class probabilities.

        Args:
            X: input batch of shape ``[N, 1, 28, 28]``.

        Returns:
            ``[N, 10]`` softmax probabilities.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4 or X.shape[1:] != INPUT_SHAPE:
            raise ValueError(
                f"Expected input of shape [N, 1, 28, 28]; got {X.shape}"
            )

        z1 = self.conv1.forward(X)
        self._z_conv1 = z1
        a1 = relu(z1)
        p1 = self.pool1.forward(a1)

        z2 = self.conv2.forward(p1)
        self._z_conv2 = z2
        a2 = relu(z2)
        p2 = self.pool2.forward(a2)

        flat = self.flatten.forward(p2)
        logits = self.head.forward(flat)
        self.logits_cache = logits
        return softmax(logits)

    def backward(self, y_true: np.ndarray) -> None:
        """Backpropagate the cross-entropy loss through the whole network.

        Populates ``conv1.dW/db``, ``conv2.dW/db``, ``head.dW/db``.
        """
        if self.logits_cache is None:
            raise RuntimeError("backward() called before forward()")

        dlogits = cross_entropy_grad_from_logits(
            self.logits_cache, y_true, reduction="mean"
        )
        dflat = self.head.backward(dlogits)
        dp2 = self.flatten.backward(dflat)
        da2 = self.pool2.backward(dp2)
        dz2 = da2 * relu_grad(self._z_conv2)
        dp1 = self.conv2.backward(dz2)
        da1 = self.pool1.backward(dp1)
        dz1 = da1 * relu_grad(self._z_conv1)
        self.conv1.backward(dz1)

    def loss_and_grads(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Forward + loss + backward; returns the scalar batch loss."""
        self.forward(X)
        loss = cross_entropy_from_logits(
            self.logits_cache, y_true, reduction="mean"
        )
        self.backward(y_true)
        return loss

    def params_and_grads(self):
        """Return ``[(W1, dW1), (b1, db1), (W2, dW2), (b2, db2), (W3, dW3), (b3, db3)]``
        for the optimizer, in a stable positional order.
        """
        pg = []
        for layer in self.trainable_layers:
            pg.append((layer.W, layer.dW))
            pg.append((layer.b, layer.db))
        return pg
