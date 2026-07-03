"""Network composition and forward/backward orchestration.

Phase 2 implemented the forward path for the fixed reference architecture:

    Input(784) -> Dense(128) + ReLU -> Dense(64) + ReLU -> Dense(10) + Softmax

Phase 4 adds the backpropagation driver. The final Softmax is fused with
the cross-entropy loss on the backward path — instead of computing the
softmax Jacobian and the CE gradient separately, the combined derivative
collapses to ``(softmax(logits) - y_true) / N`` (see ``src/loss.py``).
"""

import numpy as np

from src.activations import relu, relu_grad, softmax
from src.layers import Dense
from src.loss import cross_entropy_from_logits, cross_entropy_grad_from_logits

INPUT_DIM = 784
HIDDEN1_DIM = 128
HIDDEN2_DIM = 64
OUTPUT_DIM = 10


class Network:
    """Fixed-architecture MLP producing class probabilities.

    Architecture:
        784 -> Dense(128) + ReLU -> Dense(64) + ReLU -> Dense(10) + Softmax

    Attributes:
        layers: list of Dense layers in forward order.
        activation_cache: list of post-ReLU activations from the last forward
            pass (one per hidden layer), retained for backprop.
        logits_cache: raw logits (pre-softmax) from the last forward pass,
            retained so backward() can compute the fused softmax+CE gradient
            without recomputing the forward pass.
    """

    def __init__(self, seed=None):
        """Build the network with He-initialized dense layers.

        Args:
            seed: optional int for reproducible initialization. When provided,
                each layer is seeded deterministically but distinctly.
        """
        if seed is None:
            seeds = [None, None, None]
        else:
            seeds = [seed, seed + 1, seed + 2]

        self.layers = [
            Dense(INPUT_DIM, HIDDEN1_DIM, seed=seeds[0]),
            Dense(HIDDEN1_DIM, HIDDEN2_DIM, seed=seeds[1]),
            Dense(HIDDEN2_DIM, OUTPUT_DIM, seed=seeds[2]),
        ]
        self.activation_cache = None
        self.logits_cache = None

    def forward(self, X):
        """Run the forward pass and return class probabilities.

        Args:
            X: input batch of shape [N, 784], float.

        Returns:
            probs: array of shape [N, 10]; each row is a probability
                distribution over the 10 digit classes.
        """
        X = np.asarray(X, dtype=np.float64)

        h1 = relu(self.layers[0].forward(X))
        h2 = relu(self.layers[1].forward(h1))
        logits = self.layers[2].forward(h2)
        probs = softmax(logits)

        # Retained for backpropagation.
        self.activation_cache = [h1, h2]
        self.logits_cache = logits
        return probs

    def backward(self, y_true):
        """Backpropagate cross-entropy loss gradients through the network.

        Requires that ``forward`` was called immediately prior. Uses the
        cached logits, post-ReLU activations, and each layer's cached input
        to populate ``layers[i].dW`` and ``layers[i].db``.

        Uses mean reduction over the batch, matching ``loss(reduction="mean")``.

        Args:
            y_true: one-hot label array of shape [N, 10], matching the batch
                dimension of the most recent forward pass.
        """
        if self.logits_cache is None:
            raise RuntimeError("backward() called before forward()")

        # Fused softmax + cross-entropy derivative: (softmax(logits) - y) / N.
        dlogits = cross_entropy_grad_from_logits(
            self.logits_cache, y_true, reduction="mean"
        )

        # Output layer: Dense(64 -> 10). No activation between it and softmax.
        dh2 = self.layers[2].backward(dlogits)

        # Hidden layer 2: Dense(128 -> 64) + ReLU.
        # dz2 = dh2 * ReLU'(pre-activation of layer 1)
        dz2 = dh2 * relu_grad(self.layers[1].z_cache)
        dh1 = self.layers[1].backward(dz2)

        # Hidden layer 1: Dense(784 -> 128) + ReLU.
        dz1 = dh1 * relu_grad(self.layers[0].z_cache)
        self.layers[0].backward(dz1)

    def loss_and_grads(self, X, y_true):
        """Convenience: forward + loss + backward in one call.

        Args:
            X: input batch of shape [N, 784].
            y_true: one-hot label array of shape [N, 10].

        Returns:
            Scalar mean cross-entropy loss for the batch. After the call,
            each layer's ``dW`` and ``db`` attributes hold the parameter
            gradients w.r.t. this loss.
        """
        self.forward(X)
        loss = cross_entropy_from_logits(self.logits_cache, y_true, reduction="mean")
        self.backward(y_true)
        return loss
