"""
Numerical gradient checking utility for verifying analytical gradient implementations.

This module provides tools to estimate gradients using finite-difference methods
and compare them against analytically-derived gradients using relative error metrics.
"""

import numpy as np

# Constants for gradient checking
DEFAULT_STEP_SIZE = 1e-5
DEFAULT_TOLERANCE = 1e-7


def numerical_gradient(f, x, step_size=DEFAULT_STEP_SIZE):
    """
    Estimate the gradient of a scalar-valued function using centered finite differences.

    The gradient is computed element-wise using the central difference formula:
        ∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)

    Args:
        f: A function that takes a numpy array and returns a scalar.
        x: Point at which to estimate the gradient (numpy array).
        step_size: The finite-difference step size h. Default: 1e-5.

    Returns:
        grad: Numerical gradient estimate, shape matching input x.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x, dtype=np.float64)

    # Iterate over all elements in x
    for idx in np.ndindex(x.shape):
        # Create copies for perturbation
        x_plus = x.copy()
        x_minus = x.copy()

        # Perturb by ±h
        x_plus[idx] += step_size
        x_minus[idx] -= step_size

        # Central difference
        grad[idx] = (f(x_plus) - f(x_minus)) / (2.0 * step_size)

    return grad


def relative_error(analytic_grad, numerical_grad):
    """
    Compute the relative error between two gradients.

    Uses the robust formula: ||analytic - numerical|| / (||analytic|| + ||numerical||)

    This is more numerically stable than dividing by ||numerical|| alone when
    either gradient is small.

    Args:
        analytic_grad: Analytical gradient (numpy array or scalar).
        numerical_grad: Numerical gradient (numpy array or scalar).

    Returns:
        error: Relative error as a float in [0, ∞).
    """
    analytic = np.asarray(analytic_grad, dtype=np.float64).flatten()
    numerical = np.asarray(numerical_grad, dtype=np.float64).flatten()

    diff_norm = np.linalg.norm(analytic - numerical)
    denom_norm = np.linalg.norm(analytic) + np.linalg.norm(numerical)

    if denom_norm == 0:
        # Both gradients are zero
        return 0.0 if diff_norm == 0 else np.inf

    return diff_norm / denom_norm


def check_gradient(
    f,
    analytic_grad_fn,
    x,
    tolerance=DEFAULT_TOLERANCE,
    step_size=DEFAULT_STEP_SIZE,
    verbose=False,
):
    """
    Verify that an analytical gradient matches the numerical gradient.

    Args:
        f: The scalar-valued function to be differentiated.
        analytic_grad_fn: Function that computes the analytical gradient at x.
        x: Point at which to check the gradient.
        tolerance: Acceptance threshold for relative error. Default: 1e-7.
        step_size: Step size for numerical estimation. Default: 1e-5.
        verbose: If True, print diagnostic information.

    Returns:
        error: The relative error between analytic and numerical gradients.
        passed: Boolean indicating whether error < tolerance.
    """
    numerical_grad = numerical_gradient(f, x, step_size=step_size)
    analytic_grad = analytic_grad_fn(x)

    error = relative_error(analytic_grad, numerical_grad)
    passed = error < tolerance

    if verbose:
        print(f"Relative error: {error:.2e}")
        print(f"Tolerance: {tolerance:.2e}")
        print(f"Check passed: {passed}")

    return error, passed
