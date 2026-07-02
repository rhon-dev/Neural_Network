"""Dense (fully-connected) layer.

Phase 2 implements parameter initialization and the forward pass. The layer
caches its input and pre-activation z so that the Phase 4 backward pass can
reuse this interface without changing the forward API.
"""

import numpy as np


class Dense:
    """Fully-connected layer computing z = X @ W + b.

    Weights use He initialization (W ~ N(0, sqrt(2 / fan_in))), which is
    appropriate for the ReLU hidden layers of this network. Biases are
    initialized to zero. Initialization is seedable for reproducible tests.

    Attributes:
        W: weight matrix, shape [fan_in, fan_out].
        b: bias vector, shape [fan_out].
        input_cache: last input X seen by forward(); set to None until called.
        z_cache: last pre-activation z produced by forward(); None until called.
    """

    def __init__(self, fan_in, fan_out, seed=None):
        """Initialize layer parameters.

        Args:
            fan_in: number of input features.
            fan_out: number of output units.
            seed: optional int for reproducible weight initialization.
        """
        self.fan_in = fan_in
        self.fan_out = fan_out

        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
        self.b = np.zeros(fan_out, dtype=np.float64)

        # Caches populated during forward(); used by Phase 4 backprop.
        self.input_cache = None
        self.z_cache = None

    def forward(self, X):
        """Compute the affine transform z = X @ W + b.

        Caches the input and pre-activation for the backward pass.

        Args:
            X: input array of shape [N, fan_in].

        Returns:
            z: pre-activation array of shape [N, fan_out].
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.fan_in:
            raise ValueError(
                f"Expected input of shape [N, {self.fan_in}], got {X.shape}"
            )

        self.input_cache = X
        z = X @ self.W + self.b
        self.z_cache = z
        return z
