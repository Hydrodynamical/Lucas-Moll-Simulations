"""Particle and PDE simulations with a prescribed Lucas–Moll learning rule."""

from .config import SimulationConfig
from .simulation import run_simulation

__all__ = ["SimulationConfig", "run_simulation"]
__version__ = "0.1.0"
