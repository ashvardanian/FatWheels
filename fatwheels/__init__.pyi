"""Type stubs for FatWheels, whose functions live in compiled modules.

File: fatwheels/__init__.pyi
Author: Ash Vardanian
Date: October 29, 2026
"""

from collections.abc import Buffer
from typing import Literal, Protocol

import numpy as np

__all__ = ["capabilities", "cuda", "hip", "matmul", "reduce"]

def capabilities() -> list[str]:
    """The CPU capabilities this machine runs kernels for, the best one first."""

def reduce(a: Buffer, /, *, capability: str | None = None) -> float | int:
    """Sum of a contiguous 1D array, across every CPU core."""

def matmul(
    a: Buffer,
    b: Buffer,
    /,
    *,
    tile_size: Literal[4, 8, 16, 32, 64] = 16,
    capability: str | None = None,
) -> np.ndarray:
    """Product of two matrices with contiguous rows, tile by tile across every CPU core."""

class GPUModule(Protocol):
    """The kernels one GPU vendor's module exports, the same for NVIDIA and AMD."""

    def device_count(self) -> int:
        """The number of devices this vendor's runtime sees."""

    def reduce(self, a: Buffer, /) -> float | int:
        """Sum of a contiguous 1D array, on the first GPU."""

    def matmul(self, a: Buffer, b: Buffer, /, *, tile_size: Literal[4, 8, 16, 32] = 16) -> np.ndarray:
        """Product of two contiguous matrices, tile by tile on the first GPU."""

cuda: GPUModule | None
hip: GPUModule | None
