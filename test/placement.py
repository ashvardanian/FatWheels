"""Placement of device positions on emulated machines, whose fake device tables load what a driver would.

    pytest test/placement.py

A machine lists each device's vendor and architecture, and its table lists NVIDIA's devices before AMD's.
A build lists the codes it compiled each generation for, as CMake reports them.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

machines = {
    "rtx-h100-cdna3": [
        ("cuda", "sm_120"),
        ("cuda", "sm_120"),
        ("cuda", "sm_90"),
        ("cuda", "sm_90"),
        ("rocm", "gfx942"),
        ("rocm", "gfx942"),
    ],
    "a100-l40s": [("cuda", "sm_80"), ("cuda", "sm_89")],
    "mi250x-mi300x": [("rocm", "gfx90a"), ("rocm", "gfx942")],
}

# What the `cuda` and `rocm` presets compile each generation for.
presets = {"cuda": "80;89;120", "ampere": "80;89;120", "ada": "89;120", "rocm": "gfx942", "cdna3": "gfx942"}


@pytest.fixture(scope="session")
def placement(build) -> Path:
    return build("clang-floor") / "fatwheels_test_placement"


@pytest.fixture
def fake_machine(request: pytest.FixtureRequest, placement: Path) -> Callable[..., str]:
    """Places device ordinals on the machine named by the test's parameter, under a build's codes."""
    devices = [f"{vendor}:{architecture}" for vendor, architecture in machines[request.param]]

    def place(ordinals: list[int], codes: dict[str, str] = presets) -> str:
        generations = [f"{generation}={listed}" for generation, listed in codes.items()]
        command = [placement, *generations, *devices, *map(str, ordinals)]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

    return place


@pytest.mark.parametrize(
    "fake_machine, devices, expected",
    [
        ("rtx-h100-cdna3", [0, 1], "cuda ada"),
        ("rtx-h100-cdna3", [2, 3], "cuda ada"),  # Ada's `compute_89` PTX compiles for Hopper
        ("rtx-h100-cdna3", [1, 2], "cuda ada"),
        ("rtx-h100-cdna3", [4, 5], "rocm cdna3"),
        ("rtx-h100-cdna3", [0, 4], "refused"),
        ("rtx-h100-cdna3", [6], "refused"),
        ("rtx-h100-cdna3", [], "refused"),
        ("a100-l40s", [1], "cuda ada"),
        ("a100-l40s", [0, 1], "cuda ampere"),
        ("mi250x-mi300x", [1], "rocm cdna3"),
        ("mi250x-mi300x", [0, 1], "refused"),  # no code for gfx90a
    ],
    indirect=["fake_machine"],
)
def test_placement_resolves_one_vendor_and_generation(fake_machine, devices: list[int], expected: str):
    assert fake_machine(devices) == expected


@pytest.mark.parametrize(
    "fake_machine, codes, devices, expected",
    [
        ("rtx-h100-cdna3", {"cuda": "80;89;120", "ada": "89;120"}, [4], "refused"),  # a build without ROCm
        ("rtx-h100-cdna3", {"cuda": "80", "ada": "89-real"}, [0, 2], "cuda cuda"),
        ("a100-l40s", {"cuda": "80", "ada": "89-real"}, [1], "cuda ada"),
        ("rtx-h100-cdna3", {"cuda": "90a-real"}, [2, 3], "cuda cuda"),
        ("rtx-h100-cdna3", {"cuda": "90a"}, [0], "refused"),
    ],
    indirect=["fake_machine"],
)
def test_placement_follows_the_build_codes(fake_machine, codes: dict[str, str], devices: list[int], expected: str):
    assert fake_machine(devices, codes) == expected
