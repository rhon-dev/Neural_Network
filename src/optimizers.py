"""Parameter-update strategies (Phase 5).

Each optimizer exposes the same interface:

    optimizer.step(params_and_grads)

where ``params_and_grads`` is an iterable of ``(param, grad)`` tuples. Each
``param`` is a NumPy array updated **in place** (via ``-=``), so callers pass
references to the same arrays across steps and the optimizer accumulates any
internal state (velocity, moment estimates) keyed by positional index.

The four strategies are implemented in the order the project's Phase 5
exit criteria require:

    SGD  →  MiniBatchSGD  →  SGDMomentum  →  Adam

``MiniBatchSGD`` is not a new algorithm — plain SGD applied to a mini-batch
gradient is already mini-batch SGD. The class is exposed as a distinct
name so that the training loop is self-documenting about its intent.
"""

import numpy as np


class SGD:
    """Vanilla stochastic gradient descent: ``p -= lr * g``.

    Also serves as the update rule for mini-batch SGD — the "mini-batch"
    aspect lives in the training loop (whichever batch of examples was used
    to produce ``g``), not in the update rule itself.

    Args:
        lr: learning rate. Default: 0.01.
    """

    def __init__(self, lr=0.01):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive; got {lr}")
        self.lr = lr

    def step(self, params_and_grads):
        for p, g in params_and_grads:
            p -= self.lr * g


# ``MiniBatchSGD`` is the same algorithm as ``SGD``; the alias makes the
# intent visible at the call site. See module docstring.
MiniBatchSGD = SGD


class SGDMomentum:
    """SGD with classical (heavy-ball) momentum.

    Update:

        v_t = beta * v_{t-1} + g_t
        p_t = p_{t-1} - lr * v_t

    Args:
        lr: learning rate. Default: 0.01.
        beta: momentum coefficient in [0, 1). Default: 0.9.
    """

    def __init__(self, lr=0.01, beta=0.9):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive; got {lr}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        self.lr = lr
        self.beta = beta
        self._velocities = None

    def step(self, params_and_grads):
        pg = list(params_and_grads)
        if self._velocities is None:
            self._velocities = [np.zeros_like(p) for p, _ in pg]

        for i, (p, g) in enumerate(pg):
            self._velocities[i] = self.beta * self._velocities[i] + g
            p -= self.lr * self._velocities[i]


class Adam:
    """Adam optimizer (Kingma & Ba, 2014).

    Update (per parameter tensor):

        m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p_t   = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        lr: learning rate (a.k.a. step size). Default: 1e-3.
        beta1: exponential decay for the first-moment estimate. Default: 0.9.
        beta2: exponential decay for the second-moment estimate. Default: 0.999.
        eps: small constant added to the denominator for numerical stability.
             Default: 1e-8.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive; got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1); got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1); got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive; got {eps}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params_and_grads):
        pg = list(params_and_grads)
        if self._m is None:
            self._m = [np.zeros_like(p) for p, _ in pg]
            self._v = [np.zeros_like(p) for p, _ in pg]

        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for i, (p, g) in enumerate(pg):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
