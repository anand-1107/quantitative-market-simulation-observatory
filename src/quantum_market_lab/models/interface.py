"""Shared output convention and lightweight protocol for simulation models."""

from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class PathSimulator(Protocol):
    """A model simulator returns times and paths with paths on axis zero."""

    def simulate(self, **parameters: object) -> tuple[FloatArray, FloatArray]: ...


def validate_simulation_grid(years: float, steps: int, paths: int) -> None:
    """Validate shared finite horizon and positive integer dimensions."""
    if not np.isfinite(years) or years <= 0:
        raise ValueError("years must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths <= 0:
        raise ValueError("paths must be a positive integer")


def validate_finite_parameters(**parameters: float) -> None:
    for name, value in parameters.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
