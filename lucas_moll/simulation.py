"""Run matched particle/PDE experiments without plotting or filesystem effects."""

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .model import (
    gain_on_grid,
    learning_front,
    learning_rate,
    median_on_grid,
    particle_density,
    particle_step,
    pde_step,
)


@dataclass
class SimulationResult:
    config: SimulationConfig
    grid: np.ndarray
    times: np.ndarray
    pde_density: np.ndarray
    particle_density: np.ndarray
    pde_median: np.ndarray
    particle_median: np.ndarray
    pde_learning_front: np.ndarray
    particle_learning_front: np.ndarray
    outside_particles: np.ndarray
    final_particles: np.ndarray
    diagnostics: dict


def run_simulation(config: SimulationConfig, progress=None):
    """Store grid snapshots and final particles, rather than every N-vector.

    Both models use the same parameters, sample times and front definitions.
    Particles evolve on the real line; the PDE has reflecting boundaries.
    Particle fronts are marked undefined when a snapshot has omitted tails.
    ``progress``, if supplied, is called with (completed_steps, total_steps).
    """
    x = np.linspace(config.x_min, config.x_max, config.grid_points)
    dx = config.dx
    log_density = -0.5 * ((x - config.init_mean) / config.init_std) ** 2
    density = np.exp(log_density - log_density.max())
    density /= density.sum() * dx
    rng = np.random.default_rng(config.seed)
    positions = rng.normal(config.init_mean, config.init_std, config.particles)
    sample_steps = config.sample_steps
    samples = set(sample_steps)
    pde_snapshots, particle_snapshots, outside = [], [], []
    pde_medians, particle_medians, pde_fronts, particle_fronts = [], [], [], []
    max_mass_drift = 0.0
    clipped_mass = 0.0
    max_boundary_mass = float((density[0] + density[-1]) * dx)

    for step in range(config.steps + 1):
        # Recompute from the current density, including at observation times.
        gain = gain_on_grid(x, density * dx)
        if step in samples:
            histogram, omitted = particle_density(positions, x, dx)
            pde_snapshots.append(density.copy())
            particle_snapshots.append(histogram)
            outside.append(omitted)
            pde_medians.append(median_on_grid(x, density, dx))
            pde_fronts.append(learning_front(x, gain))
            particle_medians.append(np.nan if omitted else median_on_grid(x, histogram, dx))
            particle_fronts.append(
                np.nan if omitted else learning_front(x, gain_on_grid(x, histogram * dx))
            )
        if progress and (step % max(1, config.steps // 10) == 0 or step == config.steps):
            progress(step, config.steps)
        if step == config.steps:
            break
        rate = learning_rate(gain, config.alpha1, config.gamma)
        updated = pde_step(density, rate, config.dt, dx, config.diffusion)
        if not np.all(np.isfinite(updated)):
            raise FloatingPointError(f"Non-finite PDE density at step {step + 1}")
        max_mass_drift = max(max_mass_drift, abs(float(updated.sum() * dx) - 1))
        clipped_mass += float(-np.minimum(updated, 0).sum() * dx)
        density = np.maximum(updated, 0)
        mass = density.sum() * dx
        if mass <= 0:
            raise FloatingPointError(f"PDE mass vanished at step {step + 1}")
        density /= mass
        max_boundary_mass = max(max_boundary_mass, float((density[0] + density[-1]) * dx))
        positions = particle_step(positions, rng, config)
        if not np.all(np.isfinite(positions)):
            raise FloatingPointError(f"Non-finite particle state at step {step + 1}")

    return SimulationResult(
        config=config,
        grid=x,
        times=sample_steps * config.dt,
        pde_density=np.asarray(pde_snapshots),
        particle_density=np.asarray(particle_snapshots),
        pde_median=np.asarray(pde_medians),
        particle_median=np.asarray(particle_medians),
        pde_learning_front=np.asarray(pde_fronts),
        particle_learning_front=np.asarray(particle_fronts),
        outside_particles=np.asarray(outside),
        final_particles=positions,
        diagnostics={
            "max_pde_mass_drift_before_normalization": max_mass_drift,
            "total_pde_mass_clipped": clipped_mass,
            "max_pde_endpoint_cell_mass": max_boundary_mass,
            "max_outside_particles_at_snapshots": int(max(outside)),
        },
    )


def fit_fronts(result):
    """Fit both models on one sampled time window; these are finite-window slopes."""
    start = result.config.fit_start
    if start is None:
        return {"status": "not requested; set fit_start to fit finite-window slopes"}
    mask = result.times >= start
    times = result.times[mask]
    report = {"requested_start": start, "sample_count": int(mask.sum())}
    if len(times) < 2:
        return {**report, "status": "not fitted: fewer than two saved times in the window"}
    report.update(actual_start=float(times[0]), actual_end=float(times[-1]))
    for label, pde, particles in (
        ("median", result.pde_median, result.particle_median),
        ("learning_front", result.pde_learning_front, result.particle_learning_front),
    ):
        if not np.all(np.isfinite(pde[mask])) or not np.all(np.isfinite(particles[mask])):
            report[label] = {"status": "not fitted: undefined fronts in the common window"}
        else:
            report[label] = {
                "pde_slope": float(np.polyfit(times, pde[mask], 1)[0]),
                "particle_slope": float(np.polyfit(times, particles[mask], 1)[0]),
            }
    return report
