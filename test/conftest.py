"""Pytest options and fixtures shared by FatWheels' test modules."""

import subprocess
from pathlib import Path

import pytest

root = Path(__file__).resolve().parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-devices", action="store_true", help="also run kernels on this machine's GPUs")


@pytest.fixture(scope="session")
def build():
    """Builds a preset of the library on first use and returns its tree."""
    trees: dict[str, Path] = {}

    def tree_of(preset: str) -> Path:
        if preset not in trees:
            subprocess.run(["cmake", "--preset", preset], cwd=root, check=True, capture_output=True)
            subprocess.run(["cmake", "--build", "--preset", preset], cwd=root, check=True, capture_output=True)
            trees[preset] = root / f"build_{preset}"
        return trees[preset]

    return tree_of
