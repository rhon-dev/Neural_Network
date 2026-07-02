import numpy as np
import pytest

from src.data import batch_iterator, load_mnist, normalize_pixels, one_hot


@pytest.fixture(scope="module")
def splits():
    return load_mnist(seed=0)


def test_shapes_and_dtypes(splits):
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splits
    for X, y in [(X_tr, y_tr), (X_val, y_val), (X_te, y_te)]:
        assert X.ndim == 2 and X.shape[1] == 784
        assert y.ndim == 2 and y.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        assert X.dtype == np.float64
        assert y.dtype == np.float64


def test_split_sizes(splits):
    (X_tr, _), (X_val, _), (X_te, _) = splits
    assert X_tr.shape[0] == 50_000
    assert X_val.shape[0] == 10_000
    assert X_te.shape[0] == 10_000


def test_normalization_range(splits):
    (X_tr, _), _, _ = splits
    assert X_tr.min() >= 0.0
    assert X_tr.max() <= 1.0
    assert X_tr.max() == 1.0  # some pixel is fully saturated


def test_one_hot_validity(splits):
    for _, y in splits:
        assert np.all((y == 0.0) | (y == 1.0))
        assert np.allclose(y.sum(axis=1), 1.0)


def test_one_hot_function():
    y = one_hot(np.array([0, 3, 9]))
    expected = np.zeros((3, 10))
    expected[0, 0] = expected[1, 3] = expected[2, 9] = 1.0
    assert np.array_equal(y, expected)


def test_normalize_pixels():
    X = np.array([[0, 128, 255]], dtype=np.uint8)
    out = normalize_pixels(X)
    assert out.dtype == np.float64
    assert out.min() == 0.0 and out.max() == 1.0


def test_split_deterministic():
    (X1, _), _, _ = load_mnist(seed=0)
    (X2, _), _, _ = load_mnist(seed=0)
    assert np.array_equal(X1, X2)


def test_batch_iterator_sizes():
    X = np.arange(25).reshape(25, 1).astype(np.float64)
    y = one_hot(np.zeros(25, dtype=int))
    batches = list(batch_iterator(X, y, batch_size=10, shuffle=False))
    assert [b[0].shape[0] for b in batches] == [10, 10, 5]
    # unshuffled preserves order
    assert np.array_equal(np.vstack([b[0] for b in batches]), X)


def test_batch_iterator_shuffle_deterministic():
    X = np.arange(50).reshape(50, 1).astype(np.float64)
    y = one_hot(np.zeros(50, dtype=int))
    b1 = np.vstack([b[0] for b in batch_iterator(X, y, 8, shuffle=True, seed=42)])
    b2 = np.vstack([b[0] for b in batch_iterator(X, y, 8, shuffle=True, seed=42)])
    assert np.array_equal(b1, b2)
    assert not np.array_equal(b1, X)  # actually shuffled
    assert np.array_equal(np.sort(b1, axis=0), X)  # a permutation


def test_batch_iterator_pairs_stay_aligned():
    X = np.arange(20).reshape(20, 1).astype(np.float64)
    y = one_hot(np.arange(20) % 10)
    for xb, yb in batch_iterator(X, y, 6, shuffle=True, seed=1):
        assert np.array_equal(yb.argmax(axis=1), xb[:, 0].astype(int) % 10)


def test_batch_iterator_mismatched_rows():
    with pytest.raises(ValueError):
        list(batch_iterator(np.zeros((5, 2)), np.zeros((4, 10)), 2))
