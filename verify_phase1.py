#!/usr/bin/env python3
"""
Phase 1 Exit Criteria Verification Script

Demonstrates that the gradient checking utility correctly verifies analytical
gradients against numerical gradients on a known closed-form function with
relative error below 1e-7.
"""

import numpy as np
from src.gradient_check import (
    check_gradient,
    DEFAULT_STEP_SIZE,
    DEFAULT_TOLERANCE,
)


def main():
    print("\n" + "=" * 80)
    print(" PHASE 1: GRADIENT CHECKING UTILITY - EXIT CRITERIA VERIFICATION")
    print("=" * 80)

    # Define the test function: f(x) = sum(x^2)
    def f(x):
        """Quadratic function: f(x) = sum(x_i^2)"""
        return np.sum(x ** 2)

    # Define the analytical gradient: grad_f(x) = 2x
    def grad_f(x):
        """Analytical gradient: grad_f(x) = 2x"""
        return 2 * x

    # Test points
    test_cases = [
        ("Small array", np.array([1.0, 2.0, 3.0])),
        ("Larger array", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("Random values", np.random.randn(10)),
        ("Negative values", np.array([-1.5, -2.5, -3.5])),
        ("Mixed values", np.array([-2.0, 0.5, 3.0, -1.0])),
    ]

    print("\nTest Function: f(x) = sum(x^2)")
    print("Analytical Gradient: grad_f(x) = 2x")
    print(f"\nFinite-Difference Configuration:")
    print(f"  Step size (h): {DEFAULT_STEP_SIZE}")
    print(f"  Tolerance: {DEFAULT_TOLERANCE}")
    print(f"  Relative error formula: ||analytic - numerical|| / (||analytic|| + ||numerical||)")

    print("\n" + "-" * 80)
    print("Testing on different input configurations:")
    print("-" * 80)

    all_passed = True
    max_error = 0.0

    for name, x in test_cases:
        error, passed = check_gradient(
            f,
            grad_f,
            x,
            tolerance=DEFAULT_TOLERANCE,
            step_size=DEFAULT_STEP_SIZE,
        )
        max_error = max(max_error, error)
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"\n{name}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Relative error: {error:.3e}")
        print(f"  Status: {status}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"All tests passed: {all_passed}")
    print(f"Maximum relative error: {max_error:.3e}")
    print(f"Required tolerance: {DEFAULT_TOLERANCE:.3e}")
    print(f"Criterion satisfied: {max_error < DEFAULT_TOLERANCE}")

    if max_error < DEFAULT_TOLERANCE:
        print("\n✓ PHASE 1 EXIT CRITERION MET")
        print("Gradient checking utility successfully validates analytical gradients")
        print(f"with relative error below {DEFAULT_TOLERANCE:.0e}")
    else:
        print("\n✗ PHASE 1 EXIT CRITERION NOT MET")
        print(f"Maximum error {max_error:.3e} exceeds tolerance {DEFAULT_TOLERANCE:.3e}")
        return 1

    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
