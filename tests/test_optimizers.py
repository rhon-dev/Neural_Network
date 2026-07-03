"""Tests for Phase 5 optimizers.

Coverage:

- Update-math unit tests — one step of each optimizer against a known
  hand-computed target, so any regression in the update rule is caught.
- Convergence tests — each optimizer must reduce loss on a simple convex
  problem (quadratic in the parameter). Per-step losses are printed.
- Adam-vs-SGD comparison — on a multi-class logistic-regression problem
  fitted with the project's ``Dense`` + ``softmax`` + ``cross_entropy``
  pieces, Adam reaches a lower loss than plain SGD in the same fixed
  number of steps at the same learning rate. Both curves are printed.
"""

import numpy as np
import pytest

from src.activations import softmax
from src.layers import Dense
from src.loss import cross_entropy_from_logits, cross_entropy_grad_from_logits
from src.optimizers import SGD, Adam, MiniBatchSGD, SGDMomentum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_hot(indices, num_classes):
    y = np.zeros((len(indices), num_classes), dtype=np.float64)
    y[np.arange(len(indices)), indices] = 1.0
    return y


def _quadratic_grad(w, w_star):
    """Gradient of L(w) = 0.5 * ||w - w*||^2 is (w - w*)."""
    return w - w_star


def _quadratic_loss(w, w_star):
    return 0.5 * float(np.sum((w - w_star) ** 2))


def _run_quadratic(optimizer, num_steps, seed=0):
    """Drive a random ``w`` toward ``w_star`` under ``optimizer``.

    Returns per-step loss values.
    """
    rng = np.random.default_rng(seed)
    dim = 8
    w_star = rng.normal(size=dim)
    w = rng.normal(size=dim) * 2.0

    losses = []
    for _ in range(num_steps):
        g = _quadratic_grad(w, w_star)
        optimizer.step([(w, g)])
        losses.append(_quadratic_loss(w, w_star))
    return losses


def _run_logistic(optimizer, num_steps, seed=0):
    """Fit a Dense(D, K) -> softmax classifier on a fixed synthetic batch.

    Uses the project's Dense layer, cross-entropy loss, and fused softmax+CE
    gradient. Returns per-step loss values.
    """
    rng = np.random.default_rng(seed)
    N, D, K = 32, 6, 4
    X = rng.normal(size=(N, D))
    # Generate labels from a random ground-truth linear classifier so the
    # problem is well-defined and learnable.
    W_true = rng.normal(size=(D, K))
    y_idx = np.argmax(X @ W_true, axis=1)
    y = _one_hot(y_idx, K)

    layer = Dense(D, K, seed=seed + 1)

    losses = []
    for _ in range(num_steps):
        logits = layer.forward(X)
        loss = cross_entropy_from_logits(logits, y, reduction="mean")
        dlogits = cross_entropy_grad_from_logits(logits, y, reduction="mean")
        layer.backward(dlogits)
        optimizer.step([(layer.W, layer.dW), (layer.b, layer.db)])
        losses.append(loss)
    return losses


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_sgd_rejects_nonpositive_lr(self):
        with pytest.raises(ValueError):
            SGD(lr=0.0)
        with pytest.raises(ValueError):
            SGD(lr=-0.1)

    def test_momentum_rejects_bad_beta(self):
        with pytest.raises(ValueError):
            SGDMomentum(lr=0.01, beta=1.0)
        with pytest.raises(ValueError):
            SGDMomentum(lr=0.01, beta=-0.1)

    def test_adam_rejects_bad_hyperparams(self):
        with pytest.raises(ValueError):
            Adam(lr=-1e-3)
        with pytest.raises(ValueError):
            Adam(beta1=1.0)
        with pytest.raises(ValueError):
            Adam(beta2=-0.5)
        with pytest.raises(ValueError):
            Adam(eps=0.0)

    def test_minibatch_sgd_alias(self):
        # MiniBatchSGD is the same algorithm object as SGD.
        assert MiniBatchSGD is SGD


# ---------------------------------------------------------------------------
# Update-math unit tests
# ---------------------------------------------------------------------------


class TestSGDUpdate:
    def test_one_step_matches_hand_computed(self):
        p = np.array([1.0, 2.0, 3.0])
        g = np.array([0.5, -1.0, 0.25])
        SGD(lr=0.1).step([(p, g)])
        np.testing.assert_allclose(p, np.array([0.95, 2.1, 2.975]))

    def test_zero_gradient_leaves_params_unchanged(self):
        p = np.array([1.0, 2.0, 3.0])
        original = p.copy()
        SGD(lr=0.1).step([(p, np.zeros_like(p))])
        np.testing.assert_allclose(p, original)

    def test_in_place_update(self):
        """The array we pass in must be mutated (no fresh allocation)."""
        p = np.zeros(3)
        g = np.ones(3)
        p_id = id(p)
        SGD(lr=1.0).step([(p, g)])
        assert id(p) == p_id
        np.testing.assert_allclose(p, -np.ones(3))


class TestSGDMomentumUpdate:
    def test_first_step_matches_sgd(self):
        # With v_0 = 0: v_1 = beta*0 + g = g; step is lr*g, i.e. plain SGD.
        p = np.array([1.0, 2.0])
        g = np.array([0.5, -1.0])
        SGDMomentum(lr=0.1, beta=0.9).step([(p, g)])
        np.testing.assert_allclose(p, np.array([1.0 - 0.05, 2.0 + 0.1]))

    def test_second_step_accumulates_velocity(self):
        p = np.array([0.0, 0.0])
        g = np.array([1.0, 1.0])
        opt = SGDMomentum(lr=0.1, beta=0.9)
        opt.step([(p, g)])         # v_1 = g,           update = -0.1 * g
        opt.step([(p, g)])         # v_2 = 0.9g + g,    update = -0.1 * v_2
        # Cumulative update = -0.1 * (g + 1.9g) = -0.29 * g
        np.testing.assert_allclose(p, np.array([-0.29, -0.29]))


class TestAdamUpdate:
    def test_first_step_direction_matches_sign_of_gradient(self):
        # At t=1, bias-corrected m_hat = g, v_hat = g^2, so update magnitude
        # is lr * g / (|g| + eps) ~= lr * sign(g).
        p = np.array([0.0, 0.0])
        g = np.array([1.0, -1.0])
        Adam(lr=0.1).step([(p, g)])
        # Direction: opposite to g. Magnitude very close to lr.
        assert p[0] < 0
        assert p[1] > 0
        np.testing.assert_allclose(np.abs(p), np.array([0.1, 0.1]), atol=1e-7)

    def test_t_counter_increments_per_step(self):
        p = np.zeros(2)
        g = np.ones(2)
        opt = Adam(lr=0.01)
        assert opt._t == 0
        opt.step([(p, g)])
        assert opt._t == 1
        opt.step([(p, g)])
        assert opt._t == 2

    def test_zero_gradient_leaves_params_unchanged(self):
        p = np.array([1.0, 2.0])
        original = p.copy()
        Adam(lr=0.1).step([(p, np.zeros_like(p))])
        np.testing.assert_allclose(p, original)


# ---------------------------------------------------------------------------
# Convergence — each optimizer must reduce loss (exit criterion #1)
# ---------------------------------------------------------------------------


class TestConvergence:
    """Each optimizer reduces training loss on a small subset over N steps."""

    @pytest.mark.parametrize(
        "optimizer_factory,name",
        [
            (lambda: SGD(lr=0.1), "SGD"),
            (lambda: MiniBatchSGD(lr=0.1), "MiniBatchSGD"),
            (lambda: SGDMomentum(lr=0.05, beta=0.9), "Momentum"),
            (lambda: Adam(lr=0.1), "Adam"),
        ],
    )
    def test_reduces_quadratic_loss(self, optimizer_factory, name, capsys):
        opt = optimizer_factory()
        num_steps = 50
        losses = _run_quadratic(opt, num_steps=num_steps, seed=1)

        initial = losses[0]
        final = losses[-1]
        # Print a compact trajectory: initial, quarter, half, three-quarter, final.
        checkpoints = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, -1]
        traj = " -> ".join(f"{losses[i]:.4e}" for i in checkpoints)
        print(f"{name:<12} quadratic loss: {traj}")

        assert np.isfinite(final)
        # A meaningful reduction, not a slow drift.
        assert final < 0.1 * initial, (
            f"{name} did not reduce loss enough: {initial:.3e} -> {final:.3e}"
        )
        # Monotone-in-trend on this convex problem (compare halves).
        assert losses[num_steps // 2] < initial

    @pytest.mark.parametrize(
        "optimizer_factory,name",
        [
            (lambda: SGD(lr=0.5), "SGD"),
            (lambda: SGDMomentum(lr=0.3, beta=0.9), "Momentum"),
            (lambda: Adam(lr=0.1), "Adam"),
        ],
    )
    def test_reduces_logistic_loss(self, optimizer_factory, name, capsys):
        opt = optimizer_factory()
        num_steps = 80
        losses = _run_logistic(opt, num_steps=num_steps, seed=2)

        initial = losses[0]
        final = losses[-1]
        checkpoints = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, -1]
        traj = " -> ".join(f"{losses[i]:.4e}" for i in checkpoints)
        print(f"{name:<12} logistic  loss: {traj}")

        assert np.isfinite(final)
        assert final < initial
        # On this well-behaved problem all three should get most of the way
        # down; require at least 3x reduction.
        assert final < initial / 3.0, (
            f"{name} did not reduce logistic loss enough: "
            f"{initial:.3e} -> {final:.3e}"
        )


# ---------------------------------------------------------------------------
# Adam vs SGD comparison (exit criterion #2)
# ---------------------------------------------------------------------------


class TestAdamBeatsSGD:
    """Adam converges faster / lower than plain SGD on the same problem."""

    def test_adam_final_loss_lower_than_sgd(self, capsys):
        num_steps = 60
        sgd_losses = _run_logistic(SGD(lr=0.1), num_steps=num_steps, seed=3)
        adam_losses = _run_logistic(Adam(lr=0.1), num_steps=num_steps, seed=3)

        print(f"SGD  final loss = {sgd_losses[-1]:.4e}")
        print(f"Adam final loss = {adam_losses[-1]:.4e}")
        checkpoints = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, -1]
        print("SGD  trajectory:", " -> ".join(f"{sgd_losses[i]:.4e}" for i in checkpoints))
        print("Adam trajectory:", " -> ".join(f"{adam_losses[i]:.4e}" for i in checkpoints))

        assert adam_losses[-1] < sgd_losses[-1], (
            f"Adam did not beat SGD: adam={adam_losses[-1]:.3e} "
            f"sgd={sgd_losses[-1]:.3e}"
        )

    def test_adam_reaches_threshold_in_fewer_steps(self, capsys):
        """Steps-to-threshold comparison — a proper 'converges faster' test.

        Fit both to the same seeded problem with a fixed target loss;
        Adam should reach it strictly earlier than plain SGD.
        """
        num_steps = 200
        sgd_losses = _run_logistic(SGD(lr=0.1), num_steps=num_steps, seed=4)
        adam_losses = _run_logistic(Adam(lr=0.1), num_steps=num_steps, seed=4)

        threshold = 0.5 * sgd_losses[0]  # 2x reduction from initial

        def first_step_below(losses, thr):
            for i, v in enumerate(losses):
                if v < thr:
                    return i
            return None

        sgd_step = first_step_below(sgd_losses, threshold)
        adam_step = first_step_below(adam_losses, threshold)

        print(f"threshold           = {threshold:.4e}")
        print(f"SGD  steps-to-thr   = {sgd_step}")
        print(f"Adam steps-to-thr   = {adam_step}")

        assert adam_step is not None, "Adam never reached the threshold"
        assert sgd_step is None or adam_step < sgd_step, (
            f"Adam did not converge faster than SGD: "
            f"adam={adam_step}, sgd={sgd_step}"
        )
