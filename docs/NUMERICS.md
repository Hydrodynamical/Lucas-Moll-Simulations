# Model and numerical conventions

## Learning and innovation

Let `f(t,x)` be the distribution of knowledge states. The prescribed learning
intensity is

```math
A_f(x)=\int_x^\infty (e^{y-x}-1)f(y)\,dy,
\qquad \alpha_f(x)=\alpha_1\min\{1,A_f(x)\}^{\gamma}.
```

An agent follows `dX = sqrt(2D) dB` between meetings, with `D=1` by default.
At a meeting it samples another agent uniformly and replaces its own state
with the maximum of the two states. The empirical learning functional uses
the probability measure `(1/N) sum_j delta_{X_j}`; the self-term is zero.
Peer selection excludes self and is uniform over the other `N-1` agents.
Thus the empirical functional has denominator N while peer selection has
denominator N-1, as in the intended finite-agent convention.

The corresponding mean-field equation is

```math
\begin{aligned}
\partial_t f &= D\partial_{xx}f \\
&\quad + f(x)\int_{-\infty}^x\alpha_f(y)f(y)\,dy \\
&\quad - \alpha_f(x)f(x)\int_x^\infty f(y)\,dy.
\end{aligned}
```

No value function or optimal learning control is computed.

## PDE discretization

The grid is uniform. The Gaussian initial density is normalized on that grid.
Diffusion uses a conservative three-point Laplacian, with zero flux at the
endpoints, and time stepping is explicit Euler. Learning gain and loss use
inclusive discrete cumulative sums; their diagonal terms cancel.

Configuration validation requires

```math
\Delta t\left(2D/(\Delta x)^2+\alpha_1\right)\leq 1,
```

a sufficient positivity condition for the update when the grid density is
nonnegative and normalized. This condition also bounds the particle jump
probability by one. It does not establish time-step or spatial accuracy.
The runner records mass drift before normalization, clips negative density
values as the historical solver did, and records their total removed mass.
Non-finite states or zero total mass stop the run. Unexpected clipping or
appreciable drift calls for investigation, even if the plots look plausible.

The nonlocal functional is computed with logarithmic suffix sums. This is
O(M) in time and storage for M ordered grid points, instead of forming an
M-by-M exponential kernel. Logarithmic accumulation retains very small tail
weights on wide domains. Gains beyond floating-point range may be infinite;
the capped learning rate is still finite. Subtracting the unweighted tail can
lose relative precision for very small gains; roundoff negatives are clamped
to zero. Tests compare against a direct dense calculation at representable
scales and check tiny, exponentially amplified tails separately.

## Particle discretization

Each time step applies a Gaussian diffusion increment and then a Bernoulli
learning jump with probability `alpha * dt`, evaluated at the diffused state.
All jumps read the same pre-jump state and update simultaneously. This is a
split tau-leap scheme with at most one meeting per agent per step, not an
exact continuous-time event simulation.

Sorting followed by the same suffix calculation evaluates all empirical
learning intensities in O(N log N) time and O(N) storage. Uniform peer
selection draws an integer in `[0, N-2]` and skips over the focal index.
NumPy's `default_rng(seed)` supplies all randomness. Exact trajectories
depend on the implementation and numerical environment as well as the seed;
the saved metadata records those details.

Both solvers advance before diagnostics at the next time. Observations are
selected by integer step indices, including zero and the final step.
If more snapshots than steps are requested, duplicate indices are removed.
For S snapshots, retained state uses O(N + SM) memory. No intermediate
N-particle vectors are retained. GIF rendering can require additional memory.

## Comparing fronts

Particles are histogrammed on bins centered at the PDE grid points. Both
front calculations then use the same density pipeline:

- **Median:** linearly interpolate the grid CDF where it reaches 1/2. This
  preserves the historical grid convention; it is not `numpy.median` of the
  particle positions or a piecewise-constant, within-bin CDF interpolation.
- **Learning front:** interpolate the rightmost crossing `A_f(x)=1`, the
  boundary of the saturated learning region. A grid with no saturated region
  returns `NaN` rather than an endpoint disguised as a crossing.

The histogram is divided by the full population size. If particles fall
outside its edges, their missing mass is not renormalized and both particle
fronts for that snapshot are marked `NaN`. This is deliberately conservative
because the learning functional is extremely sensitive to the upper tail.
The out-of-grid count is checked at saved times; particles can leave and
return between observations without being counted. The PDE reflects at its
boundaries while particles evolve on the real line, so matching parameters
does not remove finite-domain error. Endpoint cell mass is recorded at every
PDE step, but small endpoint mass alone does not certify tail accuracy.

Slope fits are optional. Both models use the same observation times after
`fit_start`; at least two are required. For each observable, either both
fits use the entire selected window or neither is reported if a value is
undefined. These are finite-window linear fits without uncertainty estimates.

## Interpreting discrepancies

The exponentially weighted tail can be poorly represented in a finite
population even when the median agrees closely. That is a possible mechanism
for a learning-front discrepancy, not a conclusion established by one overlay.
Distinguish sampling variation, population-size effects, finite-domain error,
discretization, and initialization error through independent seeds, increasing
N, time/grid refinement, and domain expansion. The short default run is only
an installation and workflow demonstration.
