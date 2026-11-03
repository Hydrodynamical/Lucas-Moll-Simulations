"""Render saved runs without repeating the simulation: python -m lucas_moll.render."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS = ("#246a9e", "#c46620")


def render_run(run_directory, output_directory, *, label=None, gif=False):
    """Read the portable run artifacts and save four figures and optional GIFs."""
    source, output = Path(run_directory), Path(output_directory)
    metadata = json.loads((source / "metadata.json").read_text())
    config = metadata["config"]
    fronts = np.genfromtxt(source / "fronts.csv", delimiter=",", names=True)
    with np.load(source / "snapshots.npz", allow_pickle=False) as snapshots:
        x = snapshots["grid"]
        times = snapshots["times"]
        pde = snapshots["pde_density"]
        particles = snapshots["particle_density"]
    if pde.shape != particles.shape or pde.shape != (len(times), len(x)):
        raise ValueError("Density snapshots must agree with the saved grid and times")
    if fronts.shape != times.shape or not np.allclose(fronts["time"], times):
        raise ValueError("Front measurements must match the snapshot times")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new or empty figure directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    label = label or f"{config['particles']:,} particles"
    dx = (x[-1] - x[0]) / (len(x) - 1)
    edges = np.concatenate(([x[0] - dx / 2], x + dx / 2))
    subtitle = (
        f"{label}  |  "
        rf"$\alpha_1={config['alpha1']:g}$, $\gamma={config['gamma']:g}$, "
        rf"$\Delta t={config['dt']:g}$, seed {config['seed']}"
    )

    # Zoom the display only: include the central 99.98% of both densities at
    # all saved times and every finite front, then add padding. Data stay intact.
    limits = []
    for density in np.concatenate((pde, particles)):
        cdf = np.cumsum(density) * dx
        mass = cdf[-1]
        if mass > 0:
            indices = np.searchsorted(cdf, np.array([0.0001, 0.9999]) * mass)
            limits.extend(x[np.clip(indices, 0, len(x) - 1)])
    for name in ("pde_median", "particle_median", "pde_learning_front", "particle_learning_front"):
        limits.extend(fronts[name][np.isfinite(fronts[name])])
    left, right = min(limits), max(limits)
    padding = max(2 * dx, 0.06 * (right - left))
    x_limits = (max(x[0], left - padding), min(x[-1], right + padding))
    y_max = 1.12 * max(pde.max(), particles.max())

    def style_axis(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15, linewidth=0.7)
        ax.set_axisbelow(True)

    def save_figure(fig, name):
        fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
        fig.savefig(output / f"{name}.svg", facecolor="white")
        plt.close(fig)

    def density_panel(ax, index):
        ax.stairs(
            particles[index],
            edges,
            fill=True,
            alpha=0.3,
            color=COLORS[1],
            label="Particles on grid",
        )
        ax.plot(x, pde[index], color=COLORS[0], linewidth=2, label="PDE")
        ax.set(xlim=x_limits, ylim=(0, y_max), title=f"t = {times[index]:g}")
        style_axis(ax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    fig.suptitle("Front trajectories\n" + subtitle, fontsize=12)
    fit_start = config.get("fit_start")
    mask = times >= fit_start if fit_start is not None else np.zeros(len(times), dtype=bool)
    for ax, suffix, title in zip(axes, ("median", "learning_front"), ("Median", "Learning front")):
        values = [fronts[f"{model}_{suffix}"] for model in ("pde", "particle")]
        ax.plot(times, values[0], color=COLORS[0], linewidth=2, label="PDE")
        ax.plot(
            times,
            values[1],
            color=COLORS[1],
            linewidth=1.7,
            marker="o",
            markevery=max(1, len(times) // 20),
            markersize=3,
            label="Particles on grid",
        )
        if mask.sum() >= 2 and all(np.all(np.isfinite(v[mask])) for v in values):
            ax.axvspan(times[mask][0], times[mask][-1], color="0.5", alpha=0.09)
            coefficients = [np.polyfit(times[mask], v[mask], 1) for v in values]
            for color, coefficients_i in zip(COLORS, coefficients):
                ax.plot(
                    times[mask],
                    np.polyval(coefficients_i, times[mask]),
                    color=color,
                    linestyle="--",
                    linewidth=1,
                )
            ax.text(
                0.97,
                0.04,
                f"Fit: t = {times[mask][0]:g}–{times[mask][-1]:g}\n"
                f"PDE slope: {coefficients[0][0]:.4f}\n"
                f"Particle slope: {coefficients[1][0]:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
            )
        ax.set(xlabel="Time", ylabel="Position", title=title)
        style_axis(ax)
    axes[0].legend(frameon=False)
    save_figure(fig, "fronts")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    fig.suptitle("Particle − PDE front positions\n" + subtitle, fontsize=12)
    for ax, suffix, title in zip(axes, ("median", "learning_front"), ("Median", "Learning front")):
        ax.axhline(0, color="0.45", linewidth=1)
        ax.plot(
            times,
            fronts[f"particle_{suffix}"] - fronts[f"pde_{suffix}"],
            color=COLORS[1],
            linewidth=1.8,
        )
        ax.set(xlabel="Time", ylabel="Position difference", title=title)
        style_axis(ax)
    save_figure(fig, "front_differences")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    fig.suptitle("Final density comparison\n" + subtitle, fontsize=12)
    density_panel(ax, len(times) - 1)
    ax.set(
        xlabel="Knowledge state x",
        ylabel="Density",
        ylim=(0, 1.12 * max(pde[-1].max(), particles[-1].max())),
    )
    ax.legend(frameon=False)
    save_figure(fig, "density")

    fig, axes = plt.subplots(
        2, 3, figsize=(11, 6.5), sharex=True, sharey=True, layout="constrained"
    )
    fig.suptitle("Density evolution\n" + subtitle, fontsize=12)
    for ax, index in zip(axes.flat, np.rint(np.linspace(0, len(times) - 1, 6)).astype(int)):
        density_panel(ax, index)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Knowledge state x")
    fig.supylabel("Density")
    save_figure(fig, "density_snapshots")

    if gif:
        for name, include_pde in (("comparison", True), ("particles", False)):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.78)
            fig.suptitle(subtitle, fontsize=11, y=0.97)
            histogram = ax.stairs(
                particles[0],
                edges,
                fill=True,
                color=COLORS[1],
                alpha=0.35,
                label="Particles on grid",
            )
            (line,) = ax.plot(
                x, pde[0], color=COLORS[0], linewidth=2, label="PDE", visible=include_pde
            )
            title_artist = ax.set_title(f"t = {times[0]:g}")
            ax.set(xlim=x_limits, ylim=(0, y_max), xlabel="Knowledge state x", ylabel="Density")
            style_axis(ax)
            if include_pde:
                ax.legend(frameon=False)

            def update(index):
                histogram.set_data(particles[index])
                line.set_ydata(pde[index])
                title_artist.set_text(f"t = {times[index]:g}")
                return histogram, line, title_artist

            animation = FuncAnimation(fig, update, frames=len(times))
            animation.save(output / f"{name}.gif", writer=PillowWriter(fps=5), dpi=110)
            plt.close(fig)

    manifest = {
        "label": label,
        "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulation_metadata": metadata,
        "source_artifacts_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("metadata.json", "fronts.csv", "snapshots.npz")
        },
        "display_x_limits": [float(v) for v in x_limits],
        "display_note": "Density panels zoom to the central 99.98% of saved grid mass plus all finite fronts; full numerical grid is retained in snapshots.npz.",
        "frames": len(times) if gif else 0,
        "fps": 5 if gif else None,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved figures to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory", type=Path, help="directory containing saved numerical artifacts"
    )
    parser.add_argument("--output", type=Path, required=True, help="new or empty figure directory")
    parser.add_argument("--label", help="run label displayed on every figure")
    parser.add_argument(
        "--gif", action="store_true", help="also render comparison and particle GIFs"
    )
    args = parser.parse_args()
    render_run(args.run_directory, args.output, label=args.label, gif=args.gif)


if __name__ == "__main__":
    main()
