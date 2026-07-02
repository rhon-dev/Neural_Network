"""Network composition and forward orchestration.

Phase 2 implements the forward path only for the fixed reference architecture:

    Input(784) -> Dense(128) + ReLU -> Dense(64) + ReLU -> Dense(10) + Softmax

The backward pass (backpropagation driver) is added in Phase 4.
"""

import numpy as np

from src.activations import relu, softmax
from src.layers import Dense

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
            pass (one per hidden layer), retained for Phase 4 backprop.
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

        # Retained for Phase 4 backpropagation.
        self.activation_cache = [h1, h2]
        return probs
