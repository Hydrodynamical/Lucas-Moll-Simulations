"""Command-line runner: python -m lucas_moll --help."""

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

from .config import SimulationConfig
from .output import save_result
from .simulation import run_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a prescribed-learning PDE with interacting particles."
    )
    parser.add_argument(
        "--config", type=Path, help="JSON configuration; CLI parameters override its values"
    )
    parser.add_argument(
        "--output", type=Path, help="empty output directory (default: timestamped results folder)"
    )
    parser.add_argument("--no-plots", action="store_true", help="save only numerical results")
    parser.add_argument(
        "--gif", action="store_true", help="also save an animation of the grid snapshots"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    defaults = SimulationConfig()
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=int if type(default) is int else float,
            help=f"override {field.name} (default: {default})",
        )
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        values.update(
            {
                field.name: getattr(args, field.name)
                for field in fields(defaults)
                if getattr(args, field.name) is not None
            }
        )
        config = SimulationConfig(**values)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output must be a new or empty directory: {output}")
    if not args.quiet:
        print(
            f"{config.particles:,} particles; {config.grid_points} grid points; {config.steps:,} steps",
            flush=True,
        )

    def progress(step, total):
        print(f"  {step / total:.0%}  (t={step * config.dt:g})", flush=True)

    result = run_simulation(config, progress=None if args.quiet else progress)
    save_result(result, output, plots=not args.no_plots, gif=args.gif)
    omitted = result.diagnostics["max_outside_particles_at_snapshots"]
    if omitted:
        print(
            f"Grid omitted up to {omitted} particles; affected particle fronts are NaN. Expand the domain.",
            file=sys.stderr,
        )
    if not args.quiet:
        print(f"Saved results to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
