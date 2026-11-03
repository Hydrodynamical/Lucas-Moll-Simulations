"""Independent reference checks for the numerical changes in the active runner."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from lucas_moll import SimulationConfig, run_simulation
from lucas_moll.model import (
    gain_on_grid,
    laplacian_neumann,
    learning_front,
    learning_rate,
    particle_density,
    particle_gain,
    particle_step,
    pde_step,
    sample_other_particles,
)
from lucas_moll.simulation import fit_fronts


class GainTests(unittest.TestCase):
    def test_particle_gain_matches_direct_pairs(self):
        rng = np.random.default_rng(42)
        for positions in (np.array([2.0, 2.0, -1.0, 5.0, -1.0]), rng.normal(size=31), np.ones(8)):
            expected = np.maximum(np.expm1(positions[None, :] - positions[:, None]), 0).mean(axis=1)
            for offset in (0.0, 1_000.0, -1_000.0):
                with self.subTest(offset=offset, size=len(positions)):
                    assert_allclose(
                        particle_gain(positions + offset), expected, rtol=2e-12, atol=1e-12
                    )

    def test_grid_gain_matches_dense_kernel_with_empty_tail(self):
        x = np.linspace(-10, 20, 101)
        weights = np.exp(-(x**2) / 4)
        weights[x > 10] = 0
        weights /= weights.sum()
        expected = np.maximum(np.expm1(x[None, :] - x[:, None]), 0) @ weights
        assert_allclose(gain_on_grid(x, weights), expected, rtol=1e-12, atol=1e-12)
        assert_array_equal(gain_on_grid(x, np.zeros_like(x)), np.zeros_like(x))

    def test_wide_particle_spread_still_has_bounded_rates(self):
        gain = particle_gain(np.array([-1_000.0, 0.0, 1_000.0]))
        self.assertFalse(np.any(np.isnan(gain)))
        assert_allclose(learning_rate(gain, 0.5, 2), [0.5, 0.5, 0])

    def test_tiny_tail_contributes_after_exponential_weighting(self):
        x = np.array([-500.0, 0.0, 500.0])
        weights = np.array([1.0, 0.0, 1e-300])
        # exp(1000) itself overflows, but its weighted contribution is finite.
        self.assertAlmostEqual(np.log(gain_on_grid(x, weights)[0]), 1000 + np.log(1e-300))


class DiscretizationTests(unittest.TestCase):
    def test_diffusion_and_learning_conserve_mass(self):
        rng = np.random.default_rng(11)
        density = rng.uniform(size=50)
        dx = 0.25
        density /= density.sum() * dx
        self.assertAlmostEqual(laplacian_neumann(density, dx).sum(), 0.0, places=12)
        rate = rng.uniform(size=50)
        updated = pde_step(density, rate, dt=0.001, dx=dx, diffusion=1)
        self.assertAlmostEqual(updated.sum() * dx, 1.0, places=13)
        self.assertGreaterEqual(updated.min(), 0.0)

    def test_uniform_peer_mapping_exhausts_every_other_index(self):
        class EnumeratedDraws:
            def integers(self, low, high, size):
                self.assertion = (low, high, size)
                return np.arange(size)

        population = 7
        for focal in range(population):
            rng = EnumeratedDraws()
            peers = sample_other_particles(rng, np.full(population - 1, focal), population)
            assert_array_equal(peers, np.delete(np.arange(population), focal))
            self.assertEqual(rng.assertion, (0, population - 1, population - 1))

    def test_histogram_preserves_missing_mass(self):
        x = np.arange(5, dtype=float)
        density, omitted = particle_density(np.array([-10.0, 1.0, 2.0, 10.0]), x, 1.0)
        self.assertEqual(omitted, 2)
        self.assertEqual(density.sum(), 0.5)

    def test_learning_jumps_use_simultaneous_pre_jump_states(self):
        class FixedDraws:
            def normal(self, mean, std, size):
                return np.zeros(size)

            def random(self, size):
                return np.zeros(size)

            def integers(self, low, high, size):
                return np.array([1, 0])  # Active indices 0,1 choose peers 2,0.

        config = SimulationConfig(particles=3, diffusion=0, alpha1=1, dt=1, final_time=1)
        original = np.array([0.0, 1.0, 3.0])
        updated = particle_step(original, FixedDraws(), config)
        assert_array_equal(updated, [3.0, 1.0, 3.0])
        assert_array_equal(original, [0.0, 1.0, 3.0])

    def test_front_interpolation_and_missing_front(self):
        x = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(learning_front(x, np.array([2.0, 0.5, 0.0])), 2 / 3)
        self.assertTrue(np.isnan(learning_front(x, np.array([0.9, 0.5, 0.0]))))


class SimulationTests(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            particles=128,
            final_time=0.02,
            dt=0.002,
            grid_points=61,
            x_min=-10.0,
            x_max=20.0,
            snapshots=4,
        )
        return SimulationConfig(**(values | overrides))

    def test_pde_matches_independent_dense_euler_reference(self):
        config = self.config()
        result = run_simulation(config)
        x, dx = result.grid, config.dx
        kernel = np.maximum(np.expm1(x[None, :] - x[:, None]), 0) * dx
        density = result.pde_density[0].copy()
        # Build the reflecting diffusion matrix independently of the stencil.
        matrix = np.diag(np.full(len(x), -2.0)) + np.diag(np.ones(len(x) - 1), 1)
        matrix += np.diag(np.ones(len(x) - 1), -1)
        matrix[0, 0] = matrix[-1, -1] = -1.0
        for _ in range(config.steps):
            rate = config.alpha1 * np.minimum(1, kernel @ density) ** config.gamma
            interaction = np.array(
                [
                    density[j]
                    * (
                        sum(rate[i] * density[i] for i in range(j))
                        - rate[j] * sum(density[i] for i in range(j + 1, len(x)))
                    )
                    * dx
                    for j in range(len(x))
                ]
            )
            density += config.dt * (config.diffusion * matrix @ density / dx**2 + interaction)
            density /= density.sum() * dx
        assert_allclose(result.pde_density[-1], density, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(
            result.pde_learning_front[-1], learning_front(x, kernel @ density), places=12
        )

    def test_seeded_run_is_repeatable_and_snapshots_are_aligned(self):
        config = self.config()
        first, second = run_simulation(config), run_simulation(config)
        assert_array_equal(first.final_particles, second.final_particles)
        assert_array_equal(first.times, config.sample_steps * config.dt)
        assert_allclose(first.pde_density.sum(axis=1) * config.dx, 1.0, atol=1e-14)
        self.assertEqual(first.diagnostics["total_pde_mass_clipped"], 0.0)
        self.assertEqual(first.times[-1], config.final_time)

    def test_truncated_particle_fronts_are_not_presented_as_complete(self):
        result = run_simulation(self.config(x_min=-0.5, x_max=0.5, grid_points=3))
        self.assertTrue(np.all(result.outside_particles > 0))
        self.assertTrue(np.all(np.isnan(result.particle_median)))
        self.assertTrue(np.all(np.isnan(result.particle_learning_front)))

    def test_fits_share_time_window_and_reject_missing_values(self):
        result = run_simulation(self.config(fit_start=0.01))
        result.pde_median = 2 * result.times + 1
        result.particle_median = 3 * result.times - 1
        fits = fit_fronts(result)
        self.assertAlmostEqual(fits["median"]["pde_slope"], 2)
        self.assertAlmostEqual(fits["median"]["particle_slope"], 3)
        self.assertGreaterEqual(fits["actual_start"], 0.01)
        result.particle_median[-1] = np.nan
        self.assertIn("not fitted", fit_fronts(result)["median"]["status"])

    def test_invalid_configurations_fail_before_running(self):
        for values in (
            dict(dt=1.0),
            dict(particles=1),
            dict(gamma=0.0),
            dict(x_max=-10.0),
            dict(final_time=0.003),
            dict(snapshots=1),
            dict(alpha1=float("nan")),
            dict(seed=-1),
            dict(fit_start=0.02),
            dict(particles=2.5),
            dict(final_time=None),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(**values)


if __name__ == "__main__":
    unittest.main()
