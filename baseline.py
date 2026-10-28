"""FatWheels baselines: the Python and Numba kernels every optimized one is checked and timed against.

File: baseline.py
Author: Ash Vardanian
Date: August 9, 2024
"""

import warnings

import numpy as np
from numba import njit


def safe_njit(f):
    try:
        return njit(f)
    except Exception as e:  # noqa: BLE001 - any JIT failure falls back to plain Python
        warnings.warn(f"Numba JIT compilation failed for function {f.__name__}: {e}")
        return f


@safe_njit
def matmul(a: np.ndarray, b: np.ndarray, tile_size: int = 16) -> np.ndarray:
    """Tiled matrix multiplication equivalent to np.matmul(a, b)"""
    num_rows_a, num_cols_a = a.shape
    num_cols_b = b.shape[1]
    c = np.zeros((num_rows_a, num_cols_b), dtype=a.dtype)

    # Loop over tiles
    for i in range(0, num_rows_a, tile_size):
        for j in range(0, num_cols_b, tile_size):
            for k in range(0, num_cols_a, tile_size):
                # Compute the tile
                for ii in range(i, min(i + tile_size, num_rows_a)):
                    for jj in range(j, min(j + tile_size, num_cols_b)):
                        for kk in range(k, min(k + tile_size, num_cols_a)):
                            c[ii, jj] += a[ii, kk] * b[kk, jj]

    return c


@safe_njit
def reduce(a: np.ndarray) -> float:
    """Equivalent to np.sum(a)"""
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s
