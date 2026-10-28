import numpy as np
import warnings
from numba import njit


def safe_njit(f):
    try:
        return njit(f)
    except Exception as e:
        warnings.warn(f"Numba JIT compilation failed for function {f.__name__}: {e}")
        return f


@safe_njit
def matmul(a: np.ndarray, b: np.ndarray, tile_size: int = 16) -> np.ndarray:
    """Tiled matrix multiplication equivalent to np.matmul(a, b)"""
    m, n = a.shape
    n, k = b.shape
    c = np.zeros((m, k), dtype=a.dtype)

    # Loop over tiles
    for i in range(0, m, tile_size):
        for j in range(0, k, tile_size):
            for l in range(0, n, tile_size):

                # Compute the tile
                for ii in range(i, min(i + tile_size, m)):
                    for jj in range(j, min(j + tile_size, k)):
                        for ll in range(l, min(l + tile_size, n)):
                            c[ii, jj] += a[ii, ll] * b[ll, jj]

    return c


@safe_njit
def reduce(a: np.ndarray) -> float:
    """Equivalent to np.sum(a)"""
    s = 0.0
    for i in range(a.shape[0]):
        s += a[i]
    return s
