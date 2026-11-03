# N-particle Lucas–Moll–style jump–diffusion simulator
# - Diffusion: dX_t^k = sqrt(2) dB_t^k
# - Learning jump (tau-leap): with prob α_k(X_t, μ_t) dt, pick m≠k uniformly and set X_k ← max(X_k, X_m)
# - α_k = α1 * min(1, I_k)^γ, where I_k = ∫_{y≥x_k} e^{y-x_k} μ_t(dy) ≈ (1/N) * Σ_j e^{X_j - X_k} 1_{X_j ≥ X_k}
#
# Notes:
# - α is capped by α1, so stability is good for small dt (ensure α1*dt << 1).
# - We use a vectorized O(N log N) computation of I_k via sorting and cumulative sums.
#
# The code will:
# 1) Simulate the system for user-editable params
# 2) Plot time traces for mean and variance
# 3) Plot histograms at a few time snapshots
# 4) Save the final particle states to a CSV for inspection

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Tuple

try:
    from tqdm import tqdm  # type: ignore[import]
except ImportError:  # fallback if tqdm is unavailable
    tqdm = None

@dataclass
class SimConfig:
    N: int = 400             # number of agents
    T: float = 1.0           # final time
    dt: float = 1e-3         # time step
    alpha1: float = 1.0      # max intensity cap
    gamma: float = 1.0       # exponent on I
    seed: int = 42           # RNG seed
    diffusion_sigma: float = np.sqrt(2.0)  # sqrt(2) for your SDE
    snapshot_fractions: Sequence[float] = tuple(i / 100 for i in range(101))
    show_progress: bool = True

def compute_I_of_X(X):
    """
    Compute I_k = (1/N) sum_j exp(X_j - X_k) 1_{X_j >= X_k} for all k.
    Efficient O(N log N) using sort and reverse cumulative sums.
    """
    N = X.size
    order = np.argsort(X)            # ascending
    Xs = X[order]
    expXs = np.exp(Xs)
    # cumulative sums from right to left: S[i] = sum_{j>=i} expXs[j]
    S = np.cumsum(expXs[::-1])[::-1]
    # For an index i in sorted order, the set {j: X_j >= X_i} is j>=i
    Ik_sorted = S / expXs / N        # S_i / (e^{X_i} * N)
    # Permute back to original order
    Ik = np.empty_like(Ik_sorted)
    Ik[order] = Ik_sorted
    return Ik

def simulate(config: SimConfig):
    rng = np.random.default_rng(config.seed)
    N, T, dt = config.N, config.T, config.dt
    steps = int(np.round(T / dt))
    # Initialize X_0 with a light-tailed law (can be edited): Normal(0,1)
    X = rng.normal(0.0, 1.0, size=N)
    means, vars_, times = [], [], []
    snapshots = {}  # store hist snapshots at a few times
    fractions = sorted(set(float(f) for f in config.snapshot_fractions))
    snap_indices = set()
    for frac in fractions:
        clamped = min(max(frac, 0.0), 1.0)
        snap_time = clamped * T
        snap_indices.add(int(np.round(snap_time / dt)))

    loop = range(steps + 1)
    if config.show_progress and tqdm is not None:
        loop = tqdm(loop, total=steps + 1, desc="Simulating", unit="step")

    for n in loop:
        t = n * dt
        if n in snap_indices:
            snapshots[float(t)] = X.copy()
        means.append(X.mean())
        vars_.append(X.var())
        times.append(t)
        if n == steps:
            break

        # 1) Diffusion Euler–Maruyama step
        dB = rng.normal(0.0, np.sqrt(dt), size=N)
        X = X + config.diffusion_sigma * dB

        # 2) Compute intensities α_k
        Ik = compute_I_of_X(X)

        # old:
        # alpha = config.alpha1 * np.minimum(1.0, np.power(Ik, config.gamma))

        # new (HJB-consistent):
        k = config.gamma / (1.0 + config.gamma)
        scale = config.alpha1 * k
        alpha = config.alpha1 * np.minimum(1.0, np.power(scale * Ik, config.gamma))


        # 3) Tau-leap jump: at most one jump per particle per dt (valid for small dt)
        jump_flags = rng.uniform(0.0, 1.0, size=N) < (alpha * dt)
        if np.any(jump_flags):
            # pick one random partner m≠k for each jumping k
            partners = rng.integers(0, N, size=N)
            # ensure partners ≠ k (resample where equal)
            same = partners == np.arange(N)
            if np.any(same):
                partners[same] = (partners[same] + 1) % N
            X_jump = X.copy()
            # X_k ← max(X_k, X_m) for flagged k
            X_jump[jump_flags] = np.maximum(X[jump_flags], X[partners[jump_flags]])
            X = X_jump

    out = {
        "times": np.array(times),
        "means": np.array(means),
        "vars": np.array(vars_),
        "snapshots": snapshots,
        "final": X.copy(),
        "config": config,
    }
    return out

# ---------- Front computation helpers ----------
def compute_fronts_for_snapshot(
    X: np.ndarray, rho: float, kappa: float, alpha_prime1: float
) -> Tuple[float, float, bool]:
    N = X.size
    if N == 0:
        return np.nan, np.nan, False
    eta_m = float(np.median(X))
    Xs = np.sort(X)
    mids = (Xs[:-1] + Xs[1:]) / 2.0
    expXs = np.exp(Xs)
    cum_exp_right = np.cumsum(expXs[::-1])[::-1]
    counts_right = (len(Xs) - np.arange(len(Xs)))
    if mids.size == 0:
        A = (1.0/N) * expXs[-1]
        F = 1.0/N
        e_ell = float(Xs[-1])
        return eta_m, e_ell, True
    idx_for_mid = np.arange(1, len(Xs))
    A_vals = (1.0/N) * cum_exp_right[idx_for_mid]
    F_vals = (1.0/N) * counts_right[idx_for_mid]
    J_vals = (1.0/(rho - kappa)) * (np.exp(-mids) * A_vals - F_vals)
    target = 1.0/alpha_prime1
    diffs = J_vals - target
    crossing = np.where(diffs <= 0)[0]
    if crossing.size == 0:
        e_ell = float(mids[-1])
        ok = False
    else:
        j = crossing[0]
        if j == 0:
            e_ell = float(mids[0])
            ok = False
        else:
            x1, x2 = mids[j-1], mids[j]
            y1, y2 = diffs[j-1], diffs[j]
            e_ell = float(x1 if y2 == y1 else x1 - y1*(x2 - x1)/(y2 - y1))
            ok = True
    return eta_m, e_ell, ok

def track_fronts_over_time(result: Dict[str, Any], rho: float=2.0, alpha_prime1: float=None) -> pd.DataFrame:
    cfg = result["config"]
    sigma = float(cfg.diffusion_sigma)
    kappa = 0.5 * sigma * sigma  # κ
    if rho <= kappa:
        raise ValueError(f"Require rho>kappa; got rho={rho} ≤ kappa={kappa}")
    if alpha_prime1 is None:
        alpha_prime1 = float(cfg.alpha1)  # since α(s)=α1*s ⇒ α'(1)=α1
    rows = []
    for t in sorted(result["snapshots"].keys()):
        X = result["snapshots"][t]

        eta_m, e_ell, ok = compute_fronts_for_snapshot(X, rho=rho, kappa=kappa, alpha_prime1=alpha_prime1)
        rows.append({"time": float(t), "eta_m": eta_m, "e_ell": e_ell, "ok": bool(ok)})
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df

def learning_front_eta_from_snapshot(X, rho, kappa, alpha_prime1, gamma=1.0):
    # 1) sort and get cumulative structures
    Xs = np.sort(X)
    N = len(Xs)
    expXs = np.exp(Xs)
    # I(X_j) for j on the sorted grid:
    S = np.cumsum(expXs[::-1])[::-1]            # sum_{i>=j} e^{X_i}
    I_sorted = S / (expXs * N)                  # I at Xs[j]
    w_sorted = np.minimum(1.0, I_sorted)**gamma # w at Xs[j]
    # weighted cumulative for A_w(x) = (1/N) sum e^{X_j} w(X_j) 1_{X_j ≥ x}
    Aw_right = np.cumsum((expXs * w_sorted)[::-1])[::-1] / N
    counts_right = (N - np.arange(N)) / N
    mids = (Xs[:-1] + Xs[1:]) / 2.0
    idx = np.arange(1, N)
    Jw = (1.0/(rho - kappa)) * (np.exp(-mids) * Aw_right[idx] - counts_right[idx])
    target = 1.0 / alpha_prime1
    diff = Jw - target
    cross = np.where(diff <= 0)[0]       # ✅ first index where Jw crosses target
    if cross.size == 0:
        return float(mids[-1])
    j = cross[0]
    if j == 0:
        return float(mids[0])
    x1, x2 = mids[j-1], mids[j]
    y1, y2 = diff[j-1], diff[j]
    return float(x1 if y2 == y1 else x1 - y1*(x2 - x1)/(y2 - y1))

def track_fronts_over_time_with_learning(
    result: Dict[str, Any], rho: float = 2.0, alpha_prime1: float | None = None
) -> pd.DataFrame:
    cfg = result["config"]
    sigma = float(cfg.diffusion_sigma)
    kappa = 0.5 * sigma * sigma
    if rho <= kappa:
        raise ValueError(f"Require rho>kappa; got rho={rho} ≤ kappa={kappa}")
    if alpha_prime1 is None:
        alpha_prime1 = float(cfg.gamma * cfg.alpha1)  # α'(1) for α(s)=α1* s

    rows = []
    for t in sorted(result["snapshots"].keys()):
        X = result["snapshots"][t]
        # intrinsic front + median (you already had this)
        eta_m, e_ell, _ = compute_fronts_for_snapshot(
            X, rho=rho, kappa=kappa, alpha_prime1=alpha_prime1
        )
        # learning front (weighted with w = min(1, I)^gamma)
        eta_l = learning_front_eta_from_snapshot(
            X, rho=rho, kappa=kappa, alpha_prime1=alpha_prime1, gamma=float(cfg.gamma)
        )
        rows.append({"time": float(t), "eta_m": eta_m, "e_ell": e_ell, "eta_l": eta_l})

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

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


def save_histogram_gif(
    result,
    path="sim_histograms.gif",
    bins=100,
    fps=2,
    bin_edges=None,
    x_limits=None,
    y_limits=None,
):
    snapshots = result.get("snapshots", {})
    if not snapshots:
        raise ValueError("Simulation result has no snapshots to animate.")
    times = sorted(snapshots.keys())
    data = [snapshots[t] for t in times]
    if bin_edges is None or x_limits is None or y_limits is None:
        bin_edges, x_limits, y_limits = compute_histogram_scale(snapshots, bins)
    plot_bins_array = np.asarray(bin_edges, dtype=float)
    plot_bins: Sequence[float] = tuple(float(x) for x in plot_bins_array)
    fig, ax = plt.subplots(figsize=(6, 4))

    def update(frame_idx):
        ax.clear()
        ax.hist(data[frame_idx], bins=plot_bins, density=True)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"Histogram at t={times[frame_idx]:.3f}")
        ax.set_xlim(x_limits)
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

# # Run a baseline simulation
# cfg = SimConfig(N=400, T=1.0, dt=1e-3, alpha1=1.0, gamma=1.0, seed=7)
# result = simulate(cfg)

# Run a higher resolution simulation
cfg = SimConfig(N=10_000, T=100, dt=5e-4, alpha1=0.1, gamma=2, seed=7)
result = simulate(cfg)

# Plot time series of mean
plt.figure(figsize=(6,4))
plt.plot(result["times"], result["means"])
plt.xlabel("time")
plt.ylabel("mean X_t")
plt.title("Mean over time")
plt.tight_layout()
plt.show()

# Plot time series of variance
plt.figure(figsize=(6,4))
plt.plot(result["times"], result["vars"])
plt.xlabel("time")
plt.ylabel("Var(X_t)")
plt.title("Variance over time")
plt.tight_layout()
plt.show()

# Plot histograms at snapshots
hist_edges, x_limits, y_limits = compute_histogram_scale(result["snapshots"], bins=225)
hist_bins_array = np.asarray(hist_edges, dtype=float)
hist_bins: Sequence[float] = tuple(float(x) for x in hist_bins_array)
for t, Xsnap in result["snapshots"].items():
    plt.figure(figsize=(6,4))
    plt.hist(Xsnap, bins=hist_bins, density=True)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title(f"Histogram at t={t:.3f}")
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.tight_layout()
    plt.show()

gif_path = save_histogram_gif(
    result,
    path="sim_histograms.gif",
    bins=225,
    fps=2,
    bin_edges=hist_edges,
    x_limits=x_limits,
    y_limits=y_limits,
)

# Save final states for inspection
final_df = pd.DataFrame({"X_final": result["final"]})
final_path = "data/sim_final_states.csv"
final_df.to_csv(final_path, index=False)

print(f"Histogram GIF saved to {gif_path}")
print(f"Final states saved to {final_path}")


df_fronts = track_fronts_over_time_with_learning(result, rho=2.0)
df_fronts.to_csv("fronts_timeseries.csv", index=False)

plt.figure(); 
plt.plot(df_fronts.time, df_fronts.eta_m, label="η_m (median)")
plt.plot(df_fronts.time, df_fronts.e_ell, label="e_ℓ (intrinsic)")
plt.plot(df_fronts.time, df_fronts.eta_l, label="η_ℓ (learning, weighted)")
plt.legend(); plt.xlabel("time"); plt.ylabel("position"); plt.title("Fronts"); plt.tight_layout(); plt.show()

plt.figure(); 
plt.plot(df_fronts.time, df_fronts.e_ell - df_fronts.eta_l)
plt.xlabel("time"); plt.ylabel("e_ℓ − η_ℓ"); plt.title("Intrinsic − Learning gap"); plt.tight_layout(); plt.show()



