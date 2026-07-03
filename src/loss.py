"""Categorical cross-entropy loss and its gradient (Phase 3).

Exposes two entry points:

- ``cross_entropy_from_logits``: numerically stable, fused softmax + cross-entropy
  using the log-sum-exp trick. This is the preferred path — it avoids computing
  ``log(exp(...))`` explicitly, so it never underflows to ``-inf`` for confident
  logits and never needs epsilon clipping.

- ``cross_entropy``: convenience variant that accepts already-normalized
  probabilities from an upstream softmax. Uses epsilon clipping before ``log`` to
  keep the loss finite when a probability is exactly zero.

The analytical gradient w.r.t. the softmax input (logits) collapses to
``(softmax(logits) - y_true) / N`` under mean reduction (or ``softmax(logits) - y_true``
under sum reduction). This is the standard result — the softmax Jacobian and the
cross-entropy gradient telescope, which is also the practical reason for fusing
the two operations.
"""

import numpy as np


def _as_batched(a):
    """Return ``(a_2d, was_1d)`` — promote a 1-D vector to a batch of one."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a[np.newaxis, :], True
    return a, False


def _validate_shapes(logits_or_probs, y_true):
    if logits_or_probs.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: predictions {logits_or_probs.shape} vs "
            f"labels {y_true.shape}."
        )


def _reduce(per_sample_losses, reduction):
    """Apply ``reduction`` to a 1-D array of per-sample losses."""
    if reduction == "mean":
        return float(np.mean(per_sample_losses))
    if reduction == "sum":
        return float(np.sum(per_sample_losses))
    if reduction == "none":
        return per_sample_losses
    raise ValueError(
        f"Unknown reduction {reduction!r}; expected 'mean', 'sum', or 'none'."
    )


def cross_entropy_from_logits(logits, y_true, reduction="mean"):
    """Numerically stable categorical cross-entropy from raw logits.

    Uses the log-sum-exp identity:

        log softmax(z)_k = (z_k - m) - log( sum_j exp(z_j - m) ),  m = max_j z_j

    so that exponentials are always <= 1 and the log is always finite.

    Args:
        logits: Array of shape ``[N, K]`` (or ``[K]`` for a single sample) of
            raw class scores (pre-softmax).
        y_true: One-hot labels with the same shape as ``logits``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns:
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-sample array of shape
        ``[N]`` for ``"none"``.
    """
    z, _ = _as_batched(logits)
    y, _ = _as_batched(y_true)
    _validate_shapes(z, y)

    z_shifted = z - np.max(z, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(z_shifted), axis=1))
    log_softmax = z_shifted - log_sum_exp[:, np.newaxis]

    per_sample = -np.sum(y * log_softmax, axis=1)
    return _reduce(per_sample, reduction)


def cross_entropy(probs, y_true, reduction="mean", eps=1e-12):
    """Categorical cross-entropy from already-normalized probabilities.

    Prefer :func:`cross_entropy_from_logits` when logits are available — it is
    numerically stable without needing an epsilon clip. This variant exists for
    call sites that consume upstream ``softmax`` output directly.

    Args:
        probs: Array of shape ``[N, K]`` (or ``[K]``) with rows summing to 1.
        y_true: One-hot labels with the same shape as ``probs``.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
        eps: Lower clip applied to probabilities before ``log`` to avoid
            ``log(0) = -inf`` for confident zero predictions.

    Returns:
        Scalar loss for ``"mean"`` / ``"sum"``, or a per-sample array of shape
        ``[N]`` for ``"none"``.
    """
    p, _ = _as_batched(probs)
    y, _ = _as_batched(y_true)
    _validate_shapes(p, y)

    p_clipped = np.clip(p, eps, 1.0)
    per_sample = -np.sum(y * np.log(p_clipped), axis=1)
    return _reduce(per_sample, reduction)


def cross_entropy_grad_from_logits(logits, y_true, reduction="mean"):
    """Analytic gradient of the cross-entropy-of-softmax w.r.t. logits.

    The fused derivative collapses to a very cheap form:

        dL/dz = (softmax(z) - y_true)   for reduction="sum"
        dL/dz = (softmax(z) - y_true)/N for reduction="mean"

    Args:
        logits: Array of shape ``[N, K]`` (or ``[K]``) of raw class scores.
        y_true: One-hot labels with the same shape as ``logits``.
        reduction: Must match the reduction used to compute the loss so that
            gradients are consistent with the scalar objective. ``"none"`` is
            not accepted here because there is no single scalar to differentiate.

    Returns:
        Gradient array shaped like ``logits`` (the input's original rank is
        preserved — 1-D in, 1-D out).
    """
    z, was_1d = _as_batched(logits)
    y, _ = _as_batched(y_true)
    _validate_shapes(z, y)

    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    grad = softmax - y
    if reduction == "mean":
        grad = grad / z.shape[0]
    elif reduction != "sum":
        raise ValueError(
            f"Gradient reduction must be 'mean' or 'sum'; got {reduction!r}."
        )

    if was_1d:
        grad = grad[0]
    return grad
