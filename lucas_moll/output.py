"""Save portable numerical outputs and noninteractive comparison figures."""

import csv
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .simulation import fit_fronts


def _source_version():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def save_result(result, output, *, plots=True, gif=False):
    """Write into an empty directory. Existing runs are never overwritten."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "snapshots.npz",
        grid=result.grid,
        times=result.times,
        pde_density=result.pde_density,
        particle_density=result.particle_density,
        final_particles=result.final_particles,
    )
    columns = {
        "time": result.times,
        "pde_median": result.pde_median,
        "particle_median": result.particle_median,
        "pde_learning_front": result.pde_learning_front,
        "particle_learning_front": result.particle_learning_front,
        "outside_particles": result.outside_particles,
    }
    with (output / "fronts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    metadata = {
        "schema_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(result.config),
        "source": _source_version(),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "pillow")
            },
        },
        "diagnostics": result.diagnostics,
        "finite_window_fits": fit_fronts(result),
        "notes": [
            "Prescribed learning rule; no HJB or optimal-control equation is solved.",
            "Particle fronts are undefined when particles fall outside the histogram grid.",
            "A short run or a single seed does not establish a finite-population effect.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if plots or gif:
        _plot_result(result, output, plots=plots, gif=gif)


def _plot_result(result, output, *, plots, gif):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, pde, particles, title in (
            (axes[0], result.pde_median, result.particle_median, "Median"),
            (axes[1], result.pde_learning_front, result.particle_learning_front, "Learning front"),
        ):
            ax.plot(result.times, pde, label="PDE", color="#246a9e", linewidth=2)
            ax.plot(result.times, particles, ".--", label="Particles on grid", color="#c46620")
            ax.set(xlabel="Time", ylabel="Position", title=title)
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        fig.savefig(output / "fronts.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(result.grid, result.pde_density[-1], label="PDE", color="#246a9e", linewidth=2)
        ax.stairs(
            result.particle_density[-1],
            _bin_edges(result),
            label="Particles on grid",
            color="#c46620",
            fill=True,
            alpha=0.3,
        )
        ax.set(
            xlabel="Knowledge state x",
            ylabel="Density",
            title=f"Final state, t = {result.times[-1]:g}",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(output / "density.png", dpi=180)
        plt.close(fig)

    if gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        (pde_line,) = ax.plot(result.grid, result.pde_density[0], label="PDE", color="#246a9e")
        (particle_line,) = ax.step(
            result.grid,
            result.particle_density[0],
            where="mid",
            label="Particles on grid",
            color="#c46620",
            alpha=0.7,
        )
        ax.set(
            xlabel="Knowledge state x",
            ylabel="Density",
            xlim=(result.grid[0], result.grid[-1]),
            ylim=(0, 1.1 * max(result.pde_density.max(), result.particle_density.max())),
        )
        ax.legend(frameon=False)

        def update(index):
            pde_line.set_ydata(result.pde_density[index])
            particle_line.set_ydata(result.particle_density[index])
            ax.set_title(f"PDE and particles, t = {result.times[index]:g}")
            return pde_line, particle_line

        animation = FuncAnimation(fig, update, frames=len(result.times))
        animation.save(output / "comparison.gif", writer=PillowWriter(fps=5))
        plt.close(fig)


def _bin_edges(result):
    return np.concatenate(
        ([result.grid[0] - result.config.dx / 2], result.grid + result.config.dx / 2)
    )
