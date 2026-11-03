# ============================================================
# Boltzmann Mean Field Game:
#   - PDE simulation
#   - N-particle approximation
#   - Overlay: PDE density vs empirical histogram
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import matplotlib.animation as animation
import pandas as pd


# ============================================================
# PDE simulation of the Boltzmann Mean Field Game
# ============================================================

# -------------------------
# Parameters and grid
# -------------------------
L = 200.0          # enlarged spatial domain
N = 500
x = np.linspace(-int(L / 4), L, N)
dx = x[1] - x[0]

# Precompute exp(y - z) - 1 matrix for alpha
diff = x[None, :] - x[:, None]          # diff[j,k] = x_k - x_j
expdiff_minus1 = np.exp(diff) - 1.0

# Initial Gaussian distribution
sigma = 1.5
f = np.exp(-x**2 / (2 * sigma**2))
f /= np.sum(f) * dx                     # normalize to unit mass

# Time-stepping parameters
dt = 0.001
T_final = 50.0
num_steps = int(T_final / dt)

# Alpha parameters (shared with N-particle model)
alpha1 = 0.1
gamma = 2.0


# -------------------------
# Helper functions (PDE)
# -------------------------
def laplacian_neumann(f_arr, dx_val):
    """Second-order Laplacian with Neumann BCs on a uniform grid."""
    lap = np.zeros_like(f_arr)
    lap[1:-1] = (f_arr[:-2] - 2 * f_arr[1:-1] + f_arr[2:]) / dx_val**2
    lap[0] = (f_arr[1] - f_arr[0]) / dx_val**2
    lap[-1] = (f_arr[-2] - f_arr[-1]) / dx_val**2
    return lap


def compute_median(x_grid, f_arr, dx_val):
    """Return the median position of the probability density f on grid x."""
    cdf = np.cumsum(f_arr) * dx_val
    idx = np.searchsorted(cdf, 0.5)
    if idx == 0:
        return x_grid[0]
    if idx >= len(x_grid):
        return x_grid[-1]
    cdf_left = cdf[idx - 1]
    cdf_right = cdf[idx]
    w = (0.5 - cdf_left) / (cdf_right - cdf_left + 1e-16)
    return x_grid[idx - 1] + w * (x_grid[idx] - x_grid[idx - 1])


def compute_learning_front(x_grid, A_arr):
    """
    Learning front as the last point (moving left->right) where A(z,f) >= 1,
    i.e. the rightmost point of the saturated region {A >= 1}.
    If A never reaches 1, return np.nan.
    """
    idxs = np.where(A_arr >= 1.0)[0]
    if len(idxs) == 0:
        return np.nan
    j = idxs[-1]  # rightmost index with A >= 1
    if j == len(x_grid) - 1:
        return x_grid[j]

    A_left = A_arr[j]
    A_right = A_arr[j + 1]
    if A_right == A_left:
        return x_grid[j]

    # interpolate between x[j] (A_left >= 1) and x[j+1] (A_right < 1)
    w = (1.0 - A_left) / (A_right - A_left)
    return x_grid[j] + w * (x_grid[j + 1] - x_grid[j])


# -------------------------
# Storage for snapshots & fronts (PDE)
# -------------------------
snapshot_times = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
snapshot_indices = [int(t / dt) for t in snapshot_times]
snapshots_pde = {0.0: f.copy()}

times_pde = [0.0]
medians_pde = [compute_median(x, f, dx)]
learning_fronts_pde = []

# Initial A and learning front
kernel0 = expdiff_minus1 * (f[None, :] * dx)
upper0 = np.triu(kernel0)
A0 = np.sum(upper0, axis=1)
learning_fronts_pde.append(compute_learning_front(x, A0))


# -------------------------
# Time integration (PDE)
# -------------------------
for n in trange(1, num_steps + 1, desc="Simulating Boltzmann MFG (PDE)", unit="step"):
    # Compute alpha(z, f)
    kernel = expdiff_minus1 * (f[None, :] * dx)
    upper = np.triu(kernel)
    A = np.sum(upper, axis=1)
    alpha_base = np.minimum(1.0, A)
    alpha = alpha1 * alpha_base**gamma

    # Gain and loss terms
    prefix = np.cumsum(alpha * f) * dx
    G = f * prefix

    tail = np.cumsum(f[::-1])[::-1] * dx
    L = f * alpha * tail

    # Diffusion
    f_xx = laplacian_neumann(f, dx)

    # Explicit Euler update
    f = f + dt * (f_xx + G - L)

    # Enforce nonnegativity and renormalize
    f = np.maximum(f, 0.0)
    mass = np.sum(f) * dx
    if mass > 0:
        f /= mass

    # time and fronts
    t = n * dt
    times_pde.append(t)
    medians_pde.append(compute_median(x, f, dx))
    learning_fronts_pde.append(compute_learning_front(x, A))

    # Save snapshots
    if n in snapshot_indices:
        snapshots_pde[round(t, 5)] = f.copy()

times_pde = np.array(times_pde)
medians_pde = np.array(medians_pde)
learning_fronts_pde = np.array(learning_fronts_pde)


# -------------------------
# Fit asymptotic speeds (PDE)
# -------------------------
t0 = 15.0  # start time for asymptotic regime
mask = times_pde >= t0

coef_med = np.polyfit(times_pde[mask], medians_pde[mask], 1)           # slope, intercept
coef_lf = np.polyfit(times_pde[mask], learning_fronts_pde[mask], 1)

c_med_pde, b_med_pde = coef_med[0], coef_med[1]
c_lf_pde, b_lf_pde = coef_lf[0], coef_lf[1]


# -------------------------
# Plot PDE density snapshots
# -------------------------
plt.figure(figsize=(8, 5))
for t_snap in sorted(snapshots_pde.keys()):
    plt.plot(x, snapshots_pde[t_snap], label=f"t={t_snap:g}")
plt.title("Boltzmann MFG PDE evolution from Gaussian initial data")
plt.xlabel("x")
plt.ylabel("f(t,x)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot PDE fronts and fitted lines
plt.figure(figsize=(8, 5))
plt.plot(times_pde, medians_pde, label="median front")
plt.plot(times_pde, learning_fronts_pde, label="learning front (A>=1 boundary)")

t_fit = np.linspace(t0, T_final, 200)
plt.plot(t_fit, c_med_pde * t_fit + b_med_pde, '--', label=f"PDE median fit, c≈{c_med_pde:.3f}")
plt.plot(t_fit, c_lf_pde * t_fit + b_lf_pde, '--', label=f"PDE learning fit, c≈{c_lf_pde:.3f}")

plt.xlabel("time t")
plt.ylabel("front position")
plt.title(f"PDE fronts and asymptotic speed fits (fit from t≥{t0})")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Boltzmann MFG N-particle jump-diffusion simulator
# ============================================================

"""
N-particle approximation of the PDE

    ∂_t f = ∂_{xx}^2 f + G[f] - L[f],

with jump kernel "adopt-the-max" and learning rate

    alpha(z, f) = alpha1 * min(1, A(z,f))**gamma,
    A(z,f) = ∫_z^∞ (e^{y-z} - 1) f(y) dy.

We simulate N particles with:

    dX_t^k = sigma dB_t^k + jumps,

where jumps are "adopt-the-max" interactions at rate alpha_k depending on the
empirical analogue of A(z,f) evaluated at each particle position.
"""


# ----------------------------- Configuration ----------------------------- #

class SimConfig:
    """Plain configuration object."""
    def __init__(
        self,
        N=100_000,
        T=50.0,
        dt=1e-3,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),  # matches PDE: coefficient 1 in front of Laplacian
        alpha1=0.5,
        gamma=2.0,
        snapshot_fractions=None,
        show_progress=True,
        init_mean=0.0,
        init_std=1.5,      # matches PDE Gaussian sigma
        right_truncate=False,
        right_cap=50.0,
    ):
        self.N = N
        self.T = T
        self.dt = dt
        self.seed = seed
        self.diffusion_sigma = diffusion_sigma
        self.alpha1 = alpha1
        self.gamma = gamma
        if snapshot_fractions is None:
            # store snapshots every 1% of the total time by default
            self.snapshot_fractions = tuple(i / 100 for i in range(101))
        else:
            self.snapshot_fractions = snapshot_fractions
        self.show_progress = show_progress
        self.init_mean = init_mean
        self.init_std = init_std
        self.right_truncate = right_truncate
        self.right_cap = right_cap


# ----------------- Utility: compute A_k (PDE-compatible) ----------------- #

def compute_A_of_X(X):
    """
    Compute A_k ≈ ∫_{X_k}^∞ (e^{y - X_k} - 1) f(y) dy for the empirical measure
    f = (1/N) Σ δ_{X_j}.

    For each particle k,

        A_k = (1/N) Σ_{j: X_j >= X_k} (e^{X_j - X_k} - 1).

    This is the discrete analogue of the PDE functional A(z,f).
    Implementation is O(N log N) via sorting and reverse cumulative sums.
    """
    N = X.size
    if N == 0:
        return np.zeros_like(X)

    order = np.argsort(X)         # ascending
    Xs = X[order]

    # Numerical stability: shift by max position
    shift = float(Xs[-1])
    exp_shift = np.exp(Xs - shift)                      # e^{X_i - shift}
    S = np.cumsum(exp_shift[::-1])[::-1]                # S[i] = Σ_{j>=i} e^{X_j - shift}
    tail_counts = np.arange(N, 0, -1, dtype=float)      # # {j >= i}

    denom = np.maximum(exp_shift, np.finfo(float).tiny)
    # Σ_{j>=i} e^{X_j - X_i} = S[i] / exp_shift[i]
    E = S / denom

    # A_sorted[i] = (1/N) Σ_{j>=i} (e^{X_j - X_i} - 1)
    A_sorted = (E - tail_counts) / float(N)

    A = np.empty_like(A_sorted)
    A[order] = A_sorted
    return A

def density_from_particles_on_grid(X, x_grid, dx):
    """
    Bin particle positions X onto the PDE grid x_grid to get
    a histogram-based density f_grid(x) with total mass ≈ 1.
    """
    if X.size == 0:
        return np.zeros_like(x_grid)

    # Bin edges so that bin centers are at x_grid:
    edges = np.concatenate(([x_grid[0] - 0.5 * dx], x_grid + 0.5 * dx))
    hist, _ = np.histogram(X, bins=edges, density=False)
    N = len(X)
    f_grid = hist.astype(float) / (N * dx)
    return f_grid


def pde_style_fronts_from_particles(X, x_grid, dx, expdiff_minus1):
    """
    Given a particle snapshot X, compute median and learning front
    using the *same* PDE pipeline on a histogram density.

    Returns:
        eta_m_grid : median computed from histogram density
        z_L_grid   : learning front computed from A(x,f_grid).
    """
    # 1) histogram → density on grid
    f_grid = density_from_particles_on_grid(X, x_grid, dx)

    # 2) compute A(x,f_grid) exactly as in the PDE
    kernel = expdiff_minus1 * (f_grid[None, :] * dx)
    upper = np.triu(kernel)
    A_grid = np.sum(upper, axis=1)

    # 3) use the *same* median + learning front functions
    eta_m_grid = compute_median(x_grid, f_grid, dx)
    z_L_grid = compute_learning_front(x_grid, A_grid)

    return eta_m_grid, z_L_grid

# --------------------------- Simulator core ------------------------------ #

def simulate_particles(config: SimConfig):
    rng = np.random.default_rng(config.seed)
    N, T, dt = config.N, config.T, config.dt
    steps = int(np.round(T / dt))

    # Initial condition: Gaussian N(init_mean, init_std^2)
    X = rng.normal(config.init_mean, config.init_std, size=N)
    if config.right_truncate:
        X = np.minimum(X, config.right_cap)

    times, means, vars_ = [], [], []
    snapshots_particles = {}

    # snapshot indices
    fractions = sorted(set(float(f) for f in config.snapshot_fractions))
    snap_indices = set(int(np.round(min(max(f, 0.0), 1.0) * T / dt)) for f in fractions)

    loop = range(steps + 1)
    if config.show_progress:
        loop = tqdm(loop, total=steps + 1,
                    desc="Simulating Boltzmann MFG (N-particles)", unit="step")

    for n in loop:
        t = n * dt

        # snapshots & diagnostics
        if n in snap_indices:
            snapshots_particles[float(t)] = X.copy()
        times.append(t)
        means.append(float(np.mean(X)))
        vars_.append(float(np.var(X)))

        if n == steps:
            break

        # 1) Diffusion: Euler–Maruyama with sigma = sqrt(2)
        dB = rng.normal(0.0, np.sqrt(dt), size=N)
        X = X + config.diffusion_sigma * dB

        # 2) Learning rate alpha_k from empirical A_k (PDE functional)
        A_k = compute_A_of_X(X)
        A_clipped = np.maximum(A_k, 0.0)  # numerical safety
        alpha = config.alpha1 * np.minimum(1.0, A_clipped) ** config.gamma

        # Optional diagnostics every ~5 time units
        if n % max(1, int(5.0 / dt)) == 0:
            q90 = np.quantile(alpha, 0.90)
            mean_alpha = alpha.mean()
            frac_unsat = np.mean(A_k < 1.0)
            print(
                f"t={t:.1f}  q90(alpha)={q90:.4f}  mean alpha={mean_alpha:.4f}  "
                f"frac A<1={frac_unsat:.6f}"
            )

        # 3) Tau-leap jumps (at most one jump per particle per dt)
        jump_flags = rng.uniform(0.0, 1.0, size=N) < (alpha * dt)
        if np.any(jump_flags):
            partners = rng.integers(0, N, size=N)
            same = partners == np.arange(N)
            if np.any(same):
                partners[same] = (partners[same] + 1) % N
            X_jump = X.copy()
            X_jump[jump_flags] = np.maximum(X[jump_flags], X[partners[jump_flags]])
            X = X_jump

    out = {
        "times": np.asarray(times),
        "means": np.asarray(means),
        "vars": np.asarray(vars_),
        "snapshots": snapshots_particles,
        "final": X.copy(),
        "config": config,
    }
    return out


# ------------------------ Front computation helpers ---------------------- #

def _empirical_median(X):
    """Empirical median front, matching PDE's median definition."""
    if X.size == 0:
        return np.nan
    return float(np.median(X))


def _learning_front_from_particles(X):
    """
    Learning front as the rightmost point where A(z,f) >= 1,
    using the same A functional as in the PDE code.
    """
    N = X.size
    if N == 0:
        return np.nan

    order = np.argsort(X)
    Xs = X[order]
    A_sorted = compute_A_of_X(X)[order]

    idxs = np.where(A_sorted >= 1.0)[0]
    if len(idxs) == 0:
        return np.nan

    j = idxs[-1]         # rightmost index with A >= 1
    return float(Xs[j])  # no interpolation


def track_fronts_over_time(result_particles):
    """
    For each stored snapshot, compute:
      - eta_m: empirical median (median front)
      - z_L: learning front (A >= 1 boundary)
    """
    rows = []
    for t in sorted(result_particles["snapshots"].keys()):
        X = result_particles["snapshots"][t]
        eta_m = _empirical_median(X)
        z_L = _learning_front_from_particles(X)
        rows.append({"time": float(t), "eta_m": eta_m, "z_L": z_L})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

def track_fronts_over_time_grid(result_particles, x_grid, dx, expdiff_minus1):
    """
    For each stored particle snapshot, compute:
      - eta_m_grid: median front from histogram density on x_grid
      - z_L_grid  : learning front (A >= 1) from the same PDE pipeline.

    This puts particle fronts on exactly the same footing as PDE fronts.
    """
    rows = []
    for t in sorted(result_particles["snapshots"].keys()):
        X = result_particles["snapshots"][t]
        eta_m_grid, z_L_grid = pde_style_fronts_from_particles(
            X, x_grid, dx, expdiff_minus1
        )
        rows.append({"time": float(t), "eta_m": eta_m_grid, "z_L": z_L_grid})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# --------------------- Histogram / visualization utils ------------------- #

def compute_histogram_scale(snapshots, bins):
    if not snapshots:
        raise ValueError("Simulation result has no snapshots to analyse.")
    arrays = list(snapshots.values())
    all_data = np.concatenate(arrays)
    x_min = float(all_data.min())
    x_max = float(all_data.max())
    span = x_max - x_min
    padding = 0.05 * span if span > 0 else 1.0
    left = x_min - padding
    right = x_max + padding
    bin_edges = np.linspace(left, right, bins + 1)
    y_max = 0.0
    for arr in arrays:
        hist, _ = np.histogram(arr, bins=bin_edges, density=True)
        if hist.size:
            y_max = max(y_max, float(hist.max()))
    if y_max <= 0.0:
        y_max = 1.0
    return bin_edges, (left, right), (0.0, y_max * 1.05)


def save_histogram_gif(result_particles, path="sim_histograms.gif",
                       bins=100, fps=2, bin_edges=None, x_limits=None, y_limits=None):
    snapshots = result_particles.get("snapshots", {})
    if not snapshots:
        raise ValueError("Simulation result has no snapshots to animate.")
    times = sorted(snapshots.keys())
    data = [snapshots[t] for t in times]
    if bin_edges is None or x_limits is None or y_limits is None:
        bin_edges, x_limits, y_limits = compute_histogram_scale(snapshots, bins)
    plot_bins_array = np.asarray(bin_edges, dtype=float)
    plot_bins = tuple(float(z) for z in plot_bins_array)
    fig, ax = plt.subplots(figsize=(6, 4))

    def update(frame_idx):
        ax.clear()
        ax.hist(data[frame_idx], bins=plot_bins, density=True)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Histogram at t={times[frame_idx]:.3f}")
        if x_limits is not None:
            ax.set_xlim(x_limits)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000 / max(fps, 1),
        repeat=False,
        blit=False,
    )
    anim.save(path, writer="pillow", fps=max(fps, 1))
    plt.close(fig)
    return path


# ----------------------------- Diagnostics ------------------------------- #

def _fit_slope(ts, ys, tail_frac=0.7):
    """Simple linear fit on the late-time tail to estimate asymptotic speed."""
    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    a = int(tail_frac * len(ts))
    Xmat = np.vstack([ts[a:], np.ones(len(ts[a:]))]).T
    m, b = np.linalg.lstsq(Xmat, ys[a:], rcond=None)[0]
    return float(m)


# ------------------- PDE vs N-particle overlay animation ----------------- #

def make_pde_vs_particles_animation(
    x_grid,
    pde_snapshots,
    particle_snapshots,
    filename="pde_vs_particles.gif",
    fps=5,
):
    """
    x_grid           : 1D PDE grid
    pde_snapshots    : dict {time: f(x) array}
    particle_snapshots : dict {time: X array of particle positions}
    filename         : path for the GIF
    """

    # 1. Common times
    times_pde = set(pde_snapshots.keys())
    times_part = set(particle_snapshots.keys())
    common_times = sorted(times_pde & times_part)

    if not common_times:
        raise ValueError("No common snapshot times between PDE and particle simulations.")

    # 2. Histogram bin edges aligned with PDE grid
    dx_grid = x_grid[1] - x_grid[0]
    bin_edges = np.concatenate(([x_grid[0] - 0.5 * dx_grid], x_grid + 0.5 * dx_grid))

    # 3. Precompute y-limits
    y_max = 0.0
    for t in common_times:
        f_t = pde_snapshots[t]
        y_max = max(y_max, float(f_t.max()))

        X_t = particle_snapshots[t]
        if X_t.size > 0:
            hist, _ = np.histogram(X_t, bins=bin_edges, density=True)
            if hist.size:
                y_max = max(y_max, float(hist.max()))

    if y_max <= 0:
        y_max = 1.0

    # 4. Animation
    fig, ax = plt.subplots(figsize=(8, 4.5))

    def update(frame_idx):
        ax.clear()
        t = common_times[frame_idx]
        f_t = pde_snapshots[t]
        X_t = particle_snapshots[t]

        # PDE density
        ax.plot(x_grid, f_t, label="PDE density", linewidth=2)

        # N-particle histogram
        ax.hist(
            X_t,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            label="N-particle empirical density",
        )

        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(0, 1.05 * y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"PDE vs N-particle at t = {t:.2f}")
        ax.legend(loc="upper right")
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(common_times),
        interval=1000 / max(fps, 1),
        repeat=False,
        blit=False,
    )

    anim.save(filename, writer="pillow", fps=max(fps, 1))
    plt.close(fig)
    print(f"Overlay animation saved to {filename}")


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    # N-particle config (matched parameters)
    cfg = SimConfig(
        N=2_000_000,
        T=50.0,
        dt=1e-3,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),  # matches PDE
        alpha1=alpha1,
        gamma=gamma,
        right_truncate=False,
        right_cap=50.0,
        show_progress=True,
    )

    # Run N-particle simulation
    result_particles = simulate_particles(cfg)
    df_fronts = track_fronts_over_time_grid(
    result_particles,
    x_grid=x,
    dx=dx,
    expdiff_minus1=expdiff_minus1,
    )

    df_fronts.to_csv("fronts_timeseries_PDE_match.csv", index=False)

    # ----------------------------------------------------
    # Compare PDE vs particle fronts as functions of time
    # ----------------------------------------------------

    # Particle times and fronts (grid-based)
    times_part = df_fronts["time"].to_numpy(dtype=float)
    eta_m_part = df_fronts["eta_m"].to_numpy(dtype=float)
    z_L_part   = df_fronts["z_L"].to_numpy(dtype=float)

    # Interpolate PDE fronts at the particle snapshot times
    med_pde_interp = np.interp(times_part, times_pde, medians_pde)
    lf_pde_interp  = np.interp(times_part, times_pde, learning_fronts_pde)

    # 1) Overlay: PDE vs particle median front
    plt.figure(figsize=(8, 4.5))
    plt.plot(times_pde, medians_pde, label="PDE median", alpha=0.8)
    plt.plot(times_part, eta_m_part, "o", ms=3, label="particle median (grid-based)")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title("Median front: PDE vs particles")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Overlay: PDE vs particle learning front
    plt.figure(figsize=(8, 4.5))
    plt.plot(times_pde, learning_fronts_pde, label="PDE learning front", alpha=0.8)
    plt.plot(times_part, z_L_part, "o", ms=3, label="particle learning front (grid-based)")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title("Learning front: PDE vs particles")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Difference plots to see time dependence of the discrepancy
    plt.figure(figsize=(8, 4.5))
    plt.plot(times_part, eta_m_part - med_pde_interp)
    plt.xlabel("time")
    plt.ylabel("particle median - PDE median")
    plt.title("Median front discrepancy over time")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_part, z_L_part - lf_pde_interp)
    plt.xlabel("time")
    plt.ylabel("particle learning front - PDE learning front")
    plt.title("Learning front discrepancy over time")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.tight_layout()
    plt.show()


    # Histogram GIF (N-particle only)
    hist_edges, x_limits, y_limits = compute_histogram_scale(
        result_particles["snapshots"], bins=223
    )
    gif_path = save_histogram_gif(
        result_particles,
        path="sim_histograms_PDE_match.gif",
        bins=223,
        bin_edges=hist_edges,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    print(f"Histogram GIF saved to {gif_path}")

    # Plot N-particle fronts
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_fronts.time, df_fronts.eta_m, label="median front (particles)")
    plt.plot(df_fronts.time, df_fronts.z_L, label="learning front (particles, A ≥ 1)")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title("Fronts (N-particle, PDE-matched alpha)")
    plt.tight_layout()
    plt.show()

    # Speed estimates (N-particle)
    times_np = df_fronts.time.to_numpy(dtype=float)
    c_m = _fit_slope(times_np, df_fronts.eta_m.to_numpy(dtype=float))
    c_L = _fit_slope(times_np, df_fronts.z_L.to_numpy(dtype=float))
    print(f"Estimated late-time speeds (particles): c_m≈{c_m:.3f}, c_L≈{c_L:.3f}")
    print(f"PDE late-time speeds: c_m≈{c_med_pde:.3f}, c_L≈{c_lf_pde:.3f}")

    # PDE vs particles overlay
    make_pde_vs_particles_animation(
        x_grid=x,
        pde_snapshots=snapshots_pde,
        particle_snapshots=result_particles["snapshots"],
        filename="pde_vs_particles_overlay.gif",
        fps=4,
    )
