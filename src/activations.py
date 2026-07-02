"""Activation functions for the network (forward pass only in Phase 2).

Provides ReLU, sigmoid, and a numerically stable softmax. Activation
derivatives are implemented in Phase 4 (backpropagation).
"""

import numpy as np


def relu(z):
    """Element-wise ReLU: max(0, z).

    Args:
        z: numpy array of pre-activations.

    Returns:
        Array of the same shape with negatives clamped to zero.
    """
    return np.maximum(0.0, z)


def sigmoid(z):
    """Numerically stable element-wise logistic sigmoid.

    Uses a branch on the sign of z to avoid overflow in exp for large
    magnitude inputs:
        z >= 0:  1 / (1 + exp(-z))
        z <  0:  exp(z) / (1 + exp(z))

    Args:
        z: numpy array of pre-activations.

    Returns:
        Array of the same shape with values in (0, 1).
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)

    return out


def softmax(z):
    """Numerically stable row-wise softmax.

    Subtracts the per-row maximum before exponentiating so that the largest
    exponent is 0, preventing overflow on large inputs. Each row of the
    output is a valid probability distribution (non-negative, sums to 1).

    Args:
        z: numpy array of shape [N, K] (or [K] for a single sample) of logits.

    Returns:
        Array of the same shape; each row sums to 1 with entries in [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    single = z.ndim == 1
    if single:
        z = z[np.newaxis, :]

    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    out = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    if single:
        out = out[0]
    return out
