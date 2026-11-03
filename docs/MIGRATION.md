# Cleanup and historical compatibility

The cleanup starts from repository commit
`3df9e2b775e6ecd22846993a94f5f7e8578a0cc3`. The six versioned scripts were moved
to `legacy/` without modifying their contents. Historical data remains local
and ignored; no saved experiment was overwritten.

## Active entry point

Use `python -m lucas_moll`, rather than `python v1.py` or `python v6.py`.
The active code separates model kernels, configuration, time integration,
and output. Importing it neither runs a simulation nor opens figures.
It supports small command-line runs and records the resolved parameters,
environment, and source revision with the result.

For historical scripts, install `python -m pip install -e '.[legacy]'`
(or `python -m pip install -r requirements.txt`) and use their new paths,
for example `python legacy/v6.py`. Read the legacy README first: these
scripts retain their original defects, large defaults, interactive plots,
and fixed output names.

## Deliberate changes to numerical behavior

| Historical behavior | Active runner |
| --- | --- |
| Self-drawn peers were replaced by the next index, doubling that peer's probability | Uniform sampling over all other indices |
| PDE learning-front diagnostics used the previous step's gain with the new density's median | Both measurements use the current saved density |
| PDE speeds used times >=15 while particle speeds used roughly the last 30% | Both fits use the same saved times and explicit start time; no fit by default |
| A dense PDE kernel was formed, and particle exponentials were shifted and floored | Logarithmic suffix accumulation, verified against direct calculations |
| Histogram plots sometimes renormalized away out-of-domain mass | Full-population normalization, omitted counts, and undefined particle fronts when tails are missing |
| Many full particle snapshots were retained | Grid snapshots plus final particle states |
| Hard-coded parameters, automatic simulation on import, interactive plotting | Shared validated configuration, explicit runner, files saved without windows |

The PDE grid convention, conservative diffusion stencil, inclusive gain/loss
sums, Gaussian initialization, and split Bernoulli particle scheme are retained.
The active code is **not a bitwise replay** of historical runs: peer selection,
random-number consumption, gain arithmetic, observation timing, and default
parameters have changed. The research configuration preserves the historical
parameter values, not the original trajectory or speed report.

The README's former claim that a learning-front gap demonstrated a finite-N
effect has been qualified. Such a conclusion requires additional experiments.

## Scope of validation

Independent direct-pair and dense-PDE references, conservation checks, exact
enumeration of the peer-index mapping, seed repeatability, command-line output
checks, and short plot/GIF runs cover the refactoring. The two-million-agent,
time-50 experiment is not part of validation or CI. No new convergence or
independent-seed study is claimed.
