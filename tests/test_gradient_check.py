"""
Tests for the gradient checking utility.

Verifies that numerical gradients match analytical gradients on known closed-form
functions within specified tolerance levels.
"""

import numpy as np
import pytest
from src.gradient_check import (
    numerical_gradient,
    relative_error,
    check_gradient,
    DEFAULT_STEP_SIZE,
    DEFAULT_TOLERANCE,
)


class TestNumericalGradient:
    """Tests for finite-difference gradient estimation."""

    def test_quadratic_1d(self):
        """Test numerical gradient on f(x) = x^2."""
        def f(x):
            return np.sum(x ** 2)

        x = np.array([3.0])
        grad = numerical_gradient(f, x, step_size=1e-5)
        expected = 2 * x  # ∂/∂x(x^2) = 2x
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_quadratic_multi_d(self):
        """Test numerical gradient on f(x) = sum(x_i^2) at multiple points."""
        def f(x):
            return np.sum(x ** 2)

        x = np.array([1.0, 2.0, 3.0])
        grad = numerical_gradient(f, x, step_size=1e-5)
        expected = 2 * x
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_linear_function(self):
        """Test numerical gradient on f(x) = sum(x)."""
        def f(x):
            return np.sum(x)

        x = np.array([1.5, -2.0, 3.5])
        grad = numerical_gradient(f, x, step_size=1e-5)
        expected = np.ones_like(x)
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_cubic_function(self):
        """Test numerical gradient on f(x) = sum(x^3)."""
        def f(x):
            return np.sum(x ** 3)

        x = np.array([1.0, 2.0])
        grad = numerical_gradient(f, x, step_size=1e-5)
        expected = 3 * x ** 2  # ∂/∂x(x^3) = 3x^2
        np.testing.assert_allclose(grad, expected, rtol=1e-4)


class TestRelativeError:
    """Tests for relative error computation."""

    def test_identical_gradients(self):
        """Relative error is zero when gradients are identical."""
        grad = np.array([1.0, 2.0, 3.0])
        error = relative_error(grad, grad)
        assert error == 0.0

    def test_proportional_gradients(self):
        """Relative error with proportional gradients."""
        grad1 = np.array([1.0, 2.0])
        grad2 = np.array([2.0, 4.0])
        error = relative_error(grad1, grad2)
        # ||[1,2] - [2,4]|| = ||[-1,-2]|| = sqrt(5)
        # ||(1,2)|| + ||(2,4)|| = sqrt(5) + sqrt(20) = sqrt(5) + 2*sqrt(5) = 3*sqrt(5)
        expected = np.sqrt(5) / (3 * np.sqrt(5))
        np.testing.assert_allclose(error, expected, rtol=1e-10)

    def test_both_zero(self):
        """Relative error is zero when both gradients are zero."""
        error = relative_error(np.zeros(3), np.zeros(3))
        assert error == 0.0

    def test_one_zero_other_nonzero(self):
        """Relative error is 1.0 when one gradient is zero and the other is not."""
        # With formula: ||a - b|| / (||a|| + ||b||)
        # If a = [1, 2], b = [0, 0]: error = sqrt(5) / sqrt(5) = 1.0
        error = relative_error(np.array([1.0, 2.0]), np.zeros(2))
        assert error == 1.0

    def test_scalar_inputs(self):
        """Relative error works with scalar inputs."""
        error = relative_error(5.0, 5.0)
        assert error == 0.0

        error = relative_error(5.0, 6.0)
        assert 0 < error < 1


class TestCheckGradient:
    """Tests for the check_gradient verification function."""

    def test_quadratic_analytical_gradient(self):
        """Check that quadratic gradients verify with high accuracy."""
        def f(x):
            return np.sum(x ** 2)

        def grad_f(x):
            return 2 * x

        x = np.array([1.0, 2.0, 3.0])
        error, passed = check_gradient(
            f, grad_f, x, tolerance=DEFAULT_TOLERANCE, step_size=DEFAULT_STEP_SIZE
        )
        assert passed, f"Gradient check failed with error {error:.2e}"
        print(f"Quadratic gradient check: relative error = {error:.2e}")

    def test_linear_analytical_gradient(self):
        """Check that linear gradients verify with high accuracy."""
        def f(x):
            return np.sum(5 * x)

        def grad_f(x):
            return 5 * np.ones_like(x)

        x = np.array([1.0, -2.0, 3.5])
        error, passed = check_gradient(f, grad_f, x, tolerance=1e-6)
        assert passed, f"Gradient check failed with error {error:.2e}"
        print(f"Linear gradient check: relative error = {error:.2e}")

    def test_polynomial_analytical_gradient(self):
        """Check gradients for a cubic polynomial."""
        def f(x):
            return np.sum(x ** 3 - 2 * x ** 2 + x)

        def grad_f(x):
            return 3 * x ** 2 - 4 * x + 1

        x = np.array([0.5, 1.5, -1.0])
        error, passed = check_gradient(f, grad_f, x, tolerance=1e-6)
        assert passed, f"Gradient check failed with error {error:.2e}"
        print(f"Polynomial gradient check: relative error = {error:.2e}")

    def test_tolerance_enforcement(self):
        """Test that a too-strict tolerance correctly fails."""
        def f(x):
            return np.sum(x ** 2)

        def grad_f_wrong(x):
            # Intentionally wrong gradient
            return 3 * x

        x = np.array([1.0, 2.0])
        error, passed = check_gradient(
            f, grad_f_wrong, x, tolerance=1e-10, step_size=1e-5
        )
        assert not passed, "Expected gradient check to fail with wrong gradient"
        print(f"Wrong gradient error (should be large): {error:.2e}")


class TestPhase1ExitCriteria:
    """
    Phase 1 exit criteria: verify gradient of known closed-form test function
    with relative error below 1e-7.
    """

    def test_exit_criteria_quadratic(self):
        """
        Primary exit criterion: f(x) = sum(x^2), grad(x) = 2x.
        Relative error must be below 1e-7.
        """
        print("\n" + "=" * 70)
        print("PHASE 1 EXIT CRITERIA TEST")
        print("=" * 70)

        def f(x):
            return np.sum(x ** 2)

        def grad_f(x):
            return 2 * x

        # Test on a small array
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        error, passed = check_gradient(
            f, grad_f, x, tolerance=DEFAULT_TOLERANCE, step_size=DEFAULT_STEP_SIZE
        )

        print(f"\nTest function: f(x) = sum(x^2)")
        print(f"Analytical gradient: grad(x) = 2x")
        print(f"Test point x: {x}")
        print(f"\nFinite-difference step size: {DEFAULT_STEP_SIZE}")
        print(f"Required tolerance: {DEFAULT_TOLERANCE}")
        print(f"\nRelative error achieved: {error:.2e}")
        print(f"Criterion passed: {passed}")
        print("=" * 70)

        assert passed, f"Exit criterion failed: error {error:.2e} exceeds tolerance {DEFAULT_TOLERANCE}"

    def test_exit_criteria_random_points(self):
        """
        Extended verification: test the exit criterion at multiple random points.
        """
        print("\n" + "=" * 70)
        print("EXTENDED VERIFICATION: Random Points")
        print("=" * 70)

        def f(x):
            return np.sum(x ** 2)

        def grad_f(x):
            return 2 * x

        # Test on multiple random points
        np.random.seed(42)
        max_error = 0.0

        for trial in range(5):
            x = np.random.randn(10)
            error, passed = check_gradient(
                f, grad_f, x, tolerance=DEFAULT_TOLERANCE, step_size=DEFAULT_STEP_SIZE
            )
            max_error = max(max_error, error)
            print(f"Trial {trial + 1}: relative error = {error:.2e}, passed = {passed}")
            assert passed, f"Trial {trial + 1} failed with error {error:.2e}"

        print(f"\nMaximum relative error across all trials: {max_error:.2e}")
        print("=" * 70)


if __name__ == "__main__":
    # Run the exit criteria test explicitly
    test = TestPhase1ExitCriteria()
    test.test_exit_criteria_quadratic()
    test.test_exit_criteria_random_points()
