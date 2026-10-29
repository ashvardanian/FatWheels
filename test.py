#!/usr/bin/env python3
"""FatWheels tests: every kernel against the baseline Python and Numba code.

File: test.py
Author: Ash Vardanian
Date: August 9, 2024

This module verifies the reduction and matrix multiplication kernels against the
baseline Python/Numba code: on the CPU at every capability this machine runs,
and on every GPU vendor whose module imported.

Usage:
    uv run pytest test.py -s -x
    or
    python -m pytest test.py -s -x
"""

import numpy as np
import pytest

import fatwheels
from baseline import matmul as matmul_baseline, reduce as reduce_baseline

# Every CPU capability this machine runs, and every GPU vendor whose module imported.
backends = [f"cpu-{name}" for name in fatwheels.capabilities()]
backends += [vendor for vendor in ("cuda", "hip") if getattr(fatwheels, vendor) is not None]


def call(backend, function, *args, **kwargs):
    """Calls `function` on the CPU at the capability `backend` names, or on the GPU vendor it names."""
    device, _, capability = backend.partition("-")
    if device == "cpu":
        return getattr(fatwheels, function)(*args, capability=capability, **kwargs)
    return getattr(getattr(fatwheels, device), function)(*args, **kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("backend", backends)
def test_reduce(dtype, backend):
    """
    Test the reduction operation for different data types and backends.

    This test generates a 1D array of random values, computes the expected sum
    using the baseline Python/Numba implementation, and compares it against
    the result from the optimized C++/CUDA (or OpenMP) implementation.

    Parameters:
        dtype (np.dtype): The data type for the array elements (e.g., np.float32).
        backend (str): The backend to test, like 'cpu-haswell', 'cpu-serial', 'cuda' or 'hip'.

    Raises:
        AssertionError: If the results differ by more than the acceptable tolerance.
    """
    # Generate random data
    data = (np.random.rand(1024) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = reduce_baseline(data)

    # Get the result from the optimized implementation
    result = call(backend, "reduce", data)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32])
@pytest.mark.parametrize("backend", backends)
def test_matmul(dtype, tile_size, backend):
    """
    Test the matrix multiplication operation for different tile sizes, data types,
    and backends.

    This test generates two random 2D matrices, computes the product using the
    baseline implementation, and compares it to the product computed by the
    optimized C++/CUDA (or OpenMP) implementation.

    Parameters:
        dtype (np.dtype): The data type for the matrix elements (e.g., np.float32).
        tile_size (int): The tile size to be used for the multiplication kernel.
        backend (str): The backend to test, like 'cpu-haswell', 'cpu-serial', 'cuda' or 'hip'.

    Raises:
        AssertionError: If the output matrices differ by more than the acceptable tolerance.
    """
    # Generate random matrices
    a = (np.random.rand(256, 256) * 100).astype(dtype)
    b = (np.random.rand(256, 256) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = matmul_baseline(a, b)

    # Get the result from the optimized implementation
    result = call(backend, "matmul", a, b, tile_size=tile_size)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize("capability", fatwheels.capabilities())
def test_matmul_strided_rows(capability):
    """Row-sliced inputs are read in place, through their row strides, rather than copied or rejected."""
    a = (np.random.rand(64, 48) * 100).astype(np.float32)[::2]
    b = (np.random.rand(96, 48) * 100).astype(np.float32)[::2]
    result = fatwheels.matmul(a, b, capability=capability)
    np.testing.assert_allclose(result, matmul_baseline(np.ascontiguousarray(a), np.ascontiguousarray(b)), rtol=1e-2)


def test_matmul_rejects_overflowing_indices():
    """Matrices spanning 2^32 elements or more raise instead of wrapping their 32-bit indices."""
    huge = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.float32), shape=(70_000, 70_000), strides=(280_000, 4))
    with pytest.raises(ValueError, match="2\\^32"):
        fatwheels.matmul(huge, huge)


if __name__ == "__main__":
    pytest.main()
