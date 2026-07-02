"""Tests for the composed forward pass through the full network.

Includes the Phase 2 exit-criteria checks: valid probability distribution
output and correct [N, 10] shape.
"""

import numpy as np
from src.network import Network, INPUT_DIM, OUTPUT_DIM


class TestNetworkForward:
    """Tests for Network.forward composition and output validity."""

    def test_output_shape(self):
        net = Network(seed=0)
        X = np.zeros((16, INPUT_DIM))
        probs = net.forward(X)
        assert probs.shape == (16, OUTPUT_DIM)

    def test_rows_sum_to_one(self):
        net = Network(seed=0)
        rng = np.random.default_rng(0)
        X = rng.random((16, INPUT_DIM))
        probs = net.forward(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(16), atol=1e-12)

    def test_entries_in_unit_interval(self):
        net = Network(seed=0)
        rng = np.random.default_rng(1)
        X = rng.random((16, INPUT_DIM))
        probs = net.forward(X)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_seed_reproducible(self):
        rng = np.random.default_rng(2)
        X = rng.random((4, INPUT_DIM))
        a = Network(seed=42).forward(X)
        b = Network(seed=42).forward(X)
        np.testing.assert_array_equal(a, b)

    def test_activation_cache_populated(self):
        net = Network(seed=0)
        X = np.zeros((3, INPUT_DIM))
        net.forward(X)
        assert net.activation_cache is not None
        assert net.activation_cache[0].shape == (3, 128)
        assert net.activation_cache[1].shape == (3, 64)


class TestPhase2ExitCriteria:
    """Phase 2 exit criteria: valid probability distribution and [N, 10] shape."""

    def test_exit_criteria_probability_distribution(self):
        print("\n" + "=" * 70)
        print("PHASE 2 EXIT CRITERIA TEST")
        print("=" * 70)

        N = 64
        net = Network(seed=0)
        rng = np.random.default_rng(123)
        X = rng.random((N, INPUT_DIM))

        probs = net.forward(X)

        row_sums = probs.sum(axis=1)
        max_abs_dev = np.max(np.abs(row_sums - 1.0))
        min_entry = probs.min()
        max_entry = probs.max()

        print(f"\nBatch size N: {N}")
        print(f"Output shape: {probs.shape}")
        print(f"Max abs deviation of row sums from 1: {max_abs_dev:.3e}")
        print(f"Min entry: {min_entry:.3e}   Max entry: {max_entry:.3e}")
        print(f"All entries in [0, 1]: {bool(min_entry >= 0.0 and max_entry <= 1.0)}")

        # Criterion 2: shape is [N, 10].
        assert probs.shape == (N, OUTPUT_DIM), f"shape was {probs.shape}"

        # Criterion 1: valid probability distribution.
        np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-12)
        assert min_entry >= 0.0 and max_entry <= 1.0

        print(f"\nCriterion 1 (valid distribution): PASS")
        print(f"Criterion 2 (shape [N, 10]): PASS")
        print("=" * 70)

    def test_stability_large_preactivations(self):
        # Large-magnitude inputs must not produce NaN/inf in the output.
        net = Network(seed=0)
        X = np.full((8, INPUT_DIM), 1e3)
        probs = net.forward(X)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(8), atol=1e-12)


if __name__ == "__main__":
    test = TestPhase2ExitCriteria()
    test.test_exit_criteria_probability_distribution()
    test.test_stability_large_preactivations()
