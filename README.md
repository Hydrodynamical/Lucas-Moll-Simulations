# Lucas–Moll particle and PDE simulations

Compare an interacting population with a kinetic PDE for learning and innovation.
Each agent undergoes Brownian innovation and, at a state-dependent rate, meets
another agent and adopts the larger knowledge state. The simulation asks how
well a finite population tracks the distribution described by the PDE,
particularly in the exponentially weighted upper tail.

This is a research codebase for a **prescribed learning rule**. It does not solve
the coupled optimal-control/HJB problem in the full Lucas–Moll model. See
[the numerical model](docs/NUMERICS.md) for equations, discretization, and limitations.

## Quick start

Python 3.10 or newer, on macOS, Linux, or Windows:

```bash
git clone https://github.com/Hydrodynamical/Lucas-Moll-Simulations.git
cd Lucas-Moll-Simulations
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m lucas_moll --output results/demo
```

On Windows, create the environment with `python -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell.

The default is a small demonstration: 2,000 particles, 321 grid points, and
1,000 steps up to time 2. It saves results without opening plot windows.
An existing nonempty output directory is refused; use a new name for each run,
or omit `--output` to get a timestamped directory.

```bash
# All parameters and options
python -m lucas_moll --help

# A different seed, more particles, and an optional animation
python -m lucas_moll --particles 10000 --seed 42 --gif --output results/seed42

# A very short numerical smoke check
python -m lucas_moll --particles 100 --final-time 0.02 --snapshots 3 --no-plots
```

## Outputs

| File | Contents |
| --- | --- |
| `metadata.json` | Full configuration, source commit and dirty status, dependency versions, numerical diagnostics, and any requested slope fits |
| `fronts.csv` | Common observation times; PDE and particle median/learning fronts; particles outside the grid |
| `snapshots.npz` | Grid, times, both density histories, and final particle positions; load with `numpy.load(..., allow_pickle=False)` |
| `fronts.png` | Median and learning-front trajectories |
| `density.png` | Final PDE density and particle histogram |
| `comparison.gif` | Optional evolution animation, enabled by `--gif` |

Only grid histograms and the final particle vector are retained. Intermediate
particle identities cannot be reconstructed from the output. Fronts use the
same grid definitions in both models. A missing learning front is recorded as
`NaN`; particle fronts are also `NaN` if the histogram omits any particles.
The omitted count is reported rather than silently normalizing away the tails.

## Research configuration

[`configs/research.json`](configs/research.json) records the parameters used by
the historical `v6.py` comparison: two million particles, time 50, time step
0.001, and 500 grid points on [-50, 200]. **This is a large run**: 50,000 steps,
each sorting two million states. It is not the quick-start example or a CI test.

```bash
python -m lucas_moll --config configs/research.json --output results/research
```

JSON fields have the same names as the configuration in
[`config.py`](lucas_moll/config.py); command-line arguments override them.
Both front-speed fits use the same saved times from time 35 onward. More
generally, `--fit-start TIME` requests a finite-window slope comparison;
the demo makes no speed claim by default.

Historical plots showed close median agreement and a learning-front gap.
An individual run does not isolate a finite-population effect: independent
seeds, increasing population size, time/grid refinement, and domain checks
are still needed. The corrected runner also changes peer sampling and
diagnostic timing, so **the same seed does not reproduce the old trajectories**.
See [the migration notes](docs/MIGRATION.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `lucas_moll/config.py` | Shared parameters and validation |
| `lucas_moll/model.py` | Learning functional, PDE and particle updates, front measurements |
| `lucas_moll/simulation.py` | Matched runs and common-window fits; no plotting or file writes |
| `lucas_moll/output.py` | CSV/NPZ/JSON export and plots |
| `lucas_moll/__main__.py` | Command-line entry point |
| `configs/` | Explicit research configuration |
| `tests/` | Independent numerical references and command-line checks |
| `legacy/` | Unmodified historical versions v1–v6, including known incorrect early models |

`data/` remains ignored for local historical data; new runs go in ignored
`results/`. The original research outputs have not been regenerated or replaced.

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

CI checks Python 3.10, 3.12, and 3.13 and produces a tiny run including plots
and a GIF. Tests compare the optimized gain with a direct all-pairs calculation
and the PDE with a separate dense discretization, check uniform peer exclusion,
and verify mass conservation, diagnostic timing, repeatability, and saved output.
These are implementation checks, not a convergence study.
