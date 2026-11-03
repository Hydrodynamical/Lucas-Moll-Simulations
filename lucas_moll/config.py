"""Shared, validated parameters for the particle and PDE models."""

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Defaults are a short demonstration, not a convergence experiment."""

    particles: int = 2_000
    final_time: float = 2.0
    dt: float = 0.002
    seed: int = 7
    x_min: float = -10.0
    x_max: float = 30.0
    grid_points: int = 321
    diffusion: float = 1.0
    alpha1: float = 0.1
    gamma: float = 2.0
    init_mean: float = 0.0
    init_std: float = 1.5
    snapshots: int = 21
    fit_start: float | None = None

    def __post_init__(self):
        for name, minimum in (("particles", 2), ("grid_points", 3), ("snapshots", 2), ("seed", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "fit_start" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field.name} must be a finite number")
        if self.final_time <= 0 or self.dt <= 0:
            raise ValueError("final_time and dt must be positive")
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if not math.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("grid spacing must be finite and positive")
        if self.diffusion < 0 or self.alpha1 < 0 or self.gamma <= 0:
            raise ValueError("diffusion and alpha1 must be nonnegative; gamma must be positive")
        if self.init_std <= 0 or not self.x_min <= self.init_mean <= self.x_max:
            raise ValueError("init_std must be positive and init_mean must lie on the grid")
        ratio = self.final_time / self.dt
        if (
            not math.isfinite(ratio)
            or ratio < 1
            or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8)
        ):
            raise ValueError("final_time must be a positive integer multiple of dt")
        # Sufficient positivity bound for explicit diffusion plus learning loss.
        if self.dt * (2 * self.diffusion / self.dx**2 + self.alpha1) > 1:
            raise ValueError("dt is too large: require dt * (2*diffusion/dx**2 + alpha1) <= 1")
        if self.fit_start is not None and not 0 <= self.fit_start < self.final_time:
            raise ValueError("fit_start must satisfy 0 <= fit_start < final_time")

    @property
    def dx(self):
        return (self.x_max - self.x_min) / (self.grid_points - 1)

    @property
    def steps(self):
        return round(self.final_time / self.dt)

    @property
    def sample_steps(self):
        # Integer step indices avoid mismatched floating-point snapshot keys.
        count = min(self.snapshots, self.steps + 1)
        return np.unique(np.rint(np.linspace(0, self.steps, count)).astype(int))
