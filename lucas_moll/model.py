"""Numerical kernels. All arrays are one-dimensional and grids are uniform."""

import numpy as np


def gain_on_grid(x, weights):
    """Return A(x_i) = sum_{j>=i} (exp(x_j-x_i)-1) weights_j.

    ``x`` is increasing and ``weights`` are nonnegative probability masses
    (density times dx, or 1/N). Logarithmic suffix sums avoid an M-by-M
    kernel and retain small tail weights over wide domains. Unrepresentably
    large gains may be +inf; the bounded learning rate remains well-defined.
    """
    shifted = x - x[-1]
    with np.errstate(divide="ignore", over="ignore"):
        log_tail = np.logaddexp.accumulate((np.log(weights) + shifted)[::-1])[::-1]
        exponential_tail = np.exp(log_tail - shifted)
    tail_mass = np.cumsum(weights[::-1])[::-1]
    return np.maximum(exponential_tail - tail_mass, 0.0)


def particle_gain(positions):
    """Evaluate the empirical gain in O(N log N) time and O(N) storage."""
    if positions.size == 0:
        return np.empty(0)
    order = np.argsort(positions)
    gain = np.empty(positions.size)
    gain[order] = gain_on_grid(positions[order], np.full(positions.size, 1 / positions.size))
    return gain


def learning_rate(gain, alpha1, gamma):
    return alpha1 * np.minimum(1.0, np.maximum(gain, 0.0)) ** gamma


def laplacian_neumann(density, dx):
    """Conservative three-point diffusion with zero flux at the endpoints."""
    result = np.empty_like(density)
    result[1:-1] = (density[:-2] - 2 * density[1:-1] + density[2:]) / dx**2
    result[0] = (density[1] - density[0]) / dx**2
    result[-1] = (density[-2] - density[-1]) / dx**2
    return result


def pde_step(density, rate, dt, dx, diffusion):
    """Explicit Euler with the same inclusive gain/loss sums as v6.

    The diagonal gain and loss cancel. Returns the raw update so callers
    can measure mass drift and negative values before any normalization.
    """
    prefix = np.cumsum(rate * density) * dx
    tail = np.cumsum(density[::-1])[::-1] * dx
    return density + dt * (
        diffusion * laplacian_neumann(density, dx) + density * (prefix - rate * tail)
    )


def sample_other_particles(rng, indices, population):
    """Uniformly sample a peer other than each focal index, without rejection."""
    peers = rng.integers(0, population - 1, size=len(indices))
    return peers + (peers >= indices)


def particle_step(positions, rng, config):
    """Diffusion, then simultaneous adopt-the-max Bernoulli tau-leap jumps."""
    diffused = positions + rng.normal(
        0.0, np.sqrt(2 * config.diffusion * config.dt), size=positions.size
    )
    rates = learning_rate(particle_gain(diffused), config.alpha1, config.gamma)
    active = np.flatnonzero(rng.random(positions.size) < rates * config.dt)
    peers = sample_other_particles(rng, active, positions.size)
    updated = diffused.copy()
    updated[active] = np.maximum(diffused[active], diffused[peers])
    return updated


def particle_density(positions, x, dx):
    """Return grid density and omitted particle count; never renormalize tails."""
    edges = np.concatenate(([x[0] - dx / 2], x + dx / 2))
    counts, _ = np.histogram(positions, bins=edges)
    return counts / (positions.size * dx), int(positions.size - counts.sum())


def median_on_grid(x, density, dx):
    """Interpolate the grid CDF using the historical v6 convention."""
    cdf = np.cumsum(density) * dx
    if cdf[-1] < 0.5:
        return np.nan
    index = int(np.searchsorted(cdf, 0.5))
    if index == 0:
        return float(x[0])
    fraction = (0.5 - cdf[index - 1]) / (cdf[index] - cdf[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def learning_front(x, gain):
    """Rightmost crossing of A=1; NaN means no saturated region on the grid."""
    indices = np.flatnonzero(gain >= 1)
    if not indices.size:
        return np.nan
    index = indices[-1]
    if index == len(x) - 1:
        return float(x[index])
    # Stable interpolation even if the gain exceeds floating-point range.
    fraction = (1 - gain[index + 1]) / (gain[index] - gain[index + 1])
    return float(x[index + 1] - fraction * (x[index + 1] - x[index]))
