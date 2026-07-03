"""Dense (fully-connected) layer.

Phase 2 implements parameter initialization and the forward pass. Phase 4
adds the backward pass, which uses the input cached during forward to
compute parameter gradients (dW, db) and the upstream gradient dX.
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
        dW: gradient of loss w.r.t. W after backward(); None until called.
        db: gradient of loss w.r.t. b after backward(); None until called.
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

        # Parameter gradients populated by backward(); consumed by optimizers.
        self.dW = None
        self.db = None

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

    def backward(self, dz):
        """Backpropagate the gradient through the affine transform.

        Given ``dz = dL/dz`` where ``z = X @ W + b``, the standard matrix
        calculus identities give:

            dL/dW = X.T @ dz         (shape [fan_in, fan_out])
            dL/db = sum_n dz[n, :]   (shape [fan_out])
            dL/dX = dz @ W.T         (shape [N, fan_in])

        The batch reduction (mean vs sum) is baked into ``dz`` upstream —
        this method does not divide by ``N``. That responsibility lives with
        whoever produced ``dz`` (e.g. ``cross_entropy_grad_from_logits``
        with ``reduction="mean"`` divides once at the top of the graph).

        Requires that ``forward`` has been called; reads ``self.input_cache``.

        Args:
            dz: upstream gradient array of shape [N, fan_out].

        Returns:
            dX: gradient w.r.t. this layer's input, shape [N, fan_in].
        """
        if self.input_cache is None:
            raise RuntimeError("backward() called before forward()")

        dz = np.asarray(dz, dtype=np.float64)
        if dz.shape != (self.input_cache.shape[0], self.fan_out):
            raise ValueError(
                f"Expected dz of shape [{self.input_cache.shape[0]}, "
                f"{self.fan_out}], got {dz.shape}"
            )

        X = self.input_cache
        self.dW = X.T @ dz
        self.db = dz.sum(axis=0)
        dX = dz @ self.W.T
        return dX
