"""
Boltzmann MFG N-particle jump-diffusion simulator
=================================================

This implements an N-particle approximation of the PDE

    ∂_t f = ∂_{xx}^2 f + G[f] - L[f],

with jump kernel "adopt-the-max" and learning rate

    alpha(z, f) = alpha1 * min(1, A(z,f))**gamma,
    A(z,f) = ∫_z^∞ (e^{y-z} - 1) f(y) dy.

The PDE discretization in the reference code uses:

    alpha1 = 4.0
    gamma  = 2.0
    diffusion coefficient = 1   (⇒ sigma = sqrt(2))

Here we simulate N particles with:

    dX_t^k = sigma dB_t^k + jumps,

where jumps are "adopt-the-max" interactions at rate alpha_k depending on the
empirical analogue of A(z,f) evaluated at each particle position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # fallback if tqdm is unavailable
    tqdm = None

# ----------------------------- Configuration ----------------------------- #

class SimConfig:
    """Plain configuration object.
    Fields are set in __init__ to avoid any use of annotations.
    """
    def __init__(
        self,
        N=100_000,
        T=50.0,
        dt=1e-3,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),  # matches PDE: coefficient 1 in front of Laplacian
        alpha1=4.0,
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


# --------------------------- Simulator core ------------------------------ #

def simulate(config: SimConfig):
    rng = np.random.default_rng(config.seed)
    N, T, dt = config.N, config.T, config.dt
    steps = int(np.round(T / dt))

    # Initial condition: Gaussian N(init_mean, init_std^2)
    X = rng.normal(config.init_mean, config.init_std, size=N)

    if config.right_truncate:
        X = np.minimum(X, config.right_cap)

    times, means, vars_ = [], [], []
    snapshots = {}

    # snapshot indices
    fractions = sorted(set(float(f) for f in config.snapshot_fractions))
    snap_indices = set(int(np.round(min(max(f, 0.0), 1.0) * T / dt)) for f in fractions)

    loop = range(steps + 1)
    if config.show_progress and tqdm is not None:
        loop = tqdm(loop, total=steps + 1, desc="Simulating Boltzmann MFG N-particles", unit="step")

    for n in loop:
        t = n * dt

        # snapshots & diagnostics
        if n in snap_indices:
            snapshots[float(t)] = X.copy()
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
        "snapshots": snapshots,
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

    We approximate A(z,f) by evaluating at particle locations:
        A_k = A(X_k, empirical f),
    then define z_LF as the last position (moving left → right)
    where A_k >= 1, with optional linear interpolation to the next point.
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

    j = idxs[-1]  # rightmost index with A >= 1
    if j == N - 1:
        return float(Xs[j])

    A_left = A_sorted[j]
    A_right = A_sorted[j + 1]

    if A_right == A_left:
        return float(Xs[j])

    # interpolate between Xs[j] (A_left >= 1) and Xs[j+1] (A_right < 1 assumed)
    w = (1.0 - A_left) / (A_right - A_left)
    z_star = Xs[j] + w * (Xs[j + 1] - Xs[j])
    return float(z_star)


def track_fronts_over_time(result):
    """
    For each stored snapshot, compute:
      - eta_m: empirical median (median front)
      - z_L: learning front (A >= 1 boundary)

    Returns a DataFrame with columns: time, eta_m, z_L.
    """
    cfg = result["config"]  # not strictly needed but kept for symmetry
    rows = []
    for t in sorted(result["snapshots"].keys()):
        X = result["snapshots"][t]
        eta_m = _empirical_median(X)
        z_L = _learning_front_from_particles(X)
        rows.append({"time": float(t), "eta_m": eta_m, "z_L": z_L})
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


def save_histogram_gif(result, path="sim_histograms.gif", bins=100, fps=2,
                       bin_edges=None, x_limits=None, y_limits=None):
    snapshots = result.get("snapshots", {})
    if not snapshots:
        raise ValueError("Simulation result has no snapshots to animate.")
    times = sorted(snapshots.keys())
    data = [snapshots[t] for t in times]
    if bin_edges is None or x_limits is None or y_limits is None:
        bin_edges, x_limits, y_limits = compute_histogram_scale(snapshots, bins)
    plot_bins_array = np.asarray(bin_edges, dtype=float)
    plot_bins = tuple(float(x) for x in plot_bins_array)
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
    X = np.vstack([ts[a:], np.ones(len(ts[a:]))]).T
    m, b = np.linalg.lstsq(X, ys[a:], rcond=None)[0]
    return float(m)


# ------------------------------ Main script ------------------------------ #

if __name__ == "__main__":
    cfg = SimConfig(
        N=100_000,
        T=50.0,
        dt=1e-3,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),  # matches PDE
        alpha1=0.5,
        gamma=2.0,
        right_truncate=False,
        right_cap=50.0,
        show_progress=True,
    )

    result = simulate(cfg)
    df_fronts = track_fronts_over_time(result)
    df_fronts.to_csv("fronts_timeseries_PDE_match.csv", index=False)

    # Histogram GIF
    hist_edges, x_limits, y_limits = compute_histogram_scale(result["snapshots"], bins=223)
    gif_path = save_histogram_gif(
        result,
        path="sim_histograms_PDE_match.gif",
        bins=223,
        bin_edges=hist_edges,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    print(f"Histogram GIF saved to {gif_path}")

    # Plot fronts
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_fronts.time, df_fronts.eta_m, label="median front")
    plt.plot(df_fronts.time, df_fronts.z_L, label="learning front (A ≥ 1 boundary)")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("position")
    plt.title("Fronts (N-particle, PDE-matched alpha)")
    plt.tight_layout()
    plt.show()

    # Speed estimates
    times_np = df_fronts.time.to_numpy(dtype=float)
    c_m = _fit_slope(times_np, df_fronts.eta_m.to_numpy(dtype=float))
    c_L = _fit_slope(times_np, df_fronts.z_L.to_numpy(dtype=float))
    print(f"Estimated late-time speeds: c_m≈{c_m:.3f}, c_L≈{c_L:.3f}")
