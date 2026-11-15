"""FatWheels: one `matmul` per CPU capability and GPU generation, picked where it runs.

`fatwheels.cuda` and `fatwheels.rocm` are `None` where their modules were not built or their runtime is missing.
"""

import importlib

import numpy as np

from fatwheels import _cpu


def _shaped(raw: bytearray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).reshape(a.shape[0], b.shape[1])


def capabilities() -> list[str]:
    """The CPU capabilities this build holds and this machine runs, best first."""
    return _cpu.capabilities()


def matmul(a: np.ndarray, b: np.ndarray, capability: str | None = None) -> np.ndarray:
    """C = A * B over float32 matrices, at the best CPU capability or the one named."""
    a, b = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)
    return _shaped(_cpu.matmul(a, b, capability), a, b)


class _Vendor:
    """One GPU vendor, with the same two functions as the CPU."""

    def __init__(self, name: str):
        self._name = name

    def capabilities(self, device: int = 0) -> list[str]:
        """The generations whose kernels the driver loads on `device`, best first."""
        return _gpu.capabilities(self._name, device)

    def matmul(self, a: np.ndarray, b: np.ndarray, capability: str | None = None) -> np.ndarray:
        """C = A * B over float32 matrices on the first device, at the best generation or the one named."""
        a, b = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)
        return _shaped(_gpu.matmul(self._name, a, b, capability), a, b)


def _vendor(name: str):
    try:
        return _Vendor(name) if _gpu.capabilities(name) else None
    except ValueError:  # this build holds none of the vendor's kernels
        return None


try:
    from fatwheels import _gpu
except ImportError:  # built without GPUs
    _gpu = None

cuda = _vendor("cuda") if _gpu else None
rocm = _vendor("rocm") if _gpu else None
