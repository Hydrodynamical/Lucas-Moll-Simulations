"""
HJB-controlled Lucas-Moll N-particle jump-diffusion simulator
================================================================
This implements the *controlled* (lottery-society) dynamics:

- Diffusion: dX_t^k = sigma dB_t^k, with kappa = sigma^2/2.
- Meeting/search policy s_m(I) solves alpha'(s_m) = 1/I for a concave alpha(s) = alpha1 * s^k (0<k<1).
- Controlled Poisson jump rate: lambda(I) = alpha(s_m(I)) = alpha1 * min{1, (alpha1*k*I)^{gamma}}, with gamma = k/(1-k).
- Jump kernel: adopt-the-max (X_k ← max{X_k, X_m}), m sampled uniformly unless noted.

Learning front estimator uses the *same* policy as a weight, w = s_m(I), and solves J_w = 1/alpha'(1) = 1/(alpha1*k).

This file is self-contained. Run it directly to simulate, plot, and export a CSV of the front time series.
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
    """Plain configuration object mirroring the previous dataclass defaults.
    Fields are set in __init__ to avoid any use of annotations.
    """
    def __init__(
        self,
        N=20_000,
        T=200.0,
        dt=5e-4,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),
        rho=2.0,
        alpha1=0.25,
        gamma=1.0,
        snapshot_fractions=None,
        show_progress=True,
        init_mean=0.0,
        init_std=5.0,
        right_truncate=True,
        right_cap=10,
    ):
        self.N = N
        self.T = T
        self.dt = dt
        self.seed = seed
        self.diffusion_sigma = diffusion_sigma
        self.rho = rho
        self.alpha1 = alpha1
        self.gamma = gamma
        if snapshot_fractions is None:
            self.snapshot_fractions = tuple(i / 100 for i in range(101))
        else:
            self.snapshot_fractions = snapshot_fractions
        self.show_progress = show_progress
        self.init_mean = init_mean
        self.init_std = init_std
        self.right_truncate = right_truncate
        self.right_cap = right_cap

# ---------------------- Utility: compute I(x) fast ----------------------- #

def compute_I_of_X(X, config):
    """Compute I_k = (1/(rho - kappa)) * (1/N) * sum_j exp(X_j - X_k) * 1_{X_j >= X_k} for all k.
    O(N log N) via sorting and reverse cumulative sums. Uses a maximum shift for numerical stability.
    """
    N = X.size                                                          # number of particles
    order = np.argsort(X)                                               # ascending order indices
    Xs = X[order]                                                       # Xs = sorted X
    shift = float(Xs[-1]) if N else 0.0                                 # numerical stability shift
    exp_shift = np.exp(Xs - shift)                                      # e^{X_i - shift}
    S = np.cumsum(exp_shift[::-1])[::-1]                                # S[i] = sum_{j>=i} e^{X_j - shift}
    denom = np.maximum(exp_shift, np.finfo(float).tiny)                 # avoid division by zero
    Ik_sorted = S / (denom * N)                                         # I in sorted order
    Ik = np.empty_like(Ik_sorted)                                       # initialize unsorted I    
    Ik[order] = Ik_sorted                                               # reorder to original indices  
    return Ik/(config.rho - ((config.diffusion_sigma**2)/2))            # scale by 1/(rho - kappa)

# --------------------------- Simulator core ------------------------------ #

def simulate(config):
    # Initialize number of random seed, N, T, dt and define number of steps
    rng = np.random.default_rng(config.seed)        # set random seed
    N, T, dt = config.N, config.T, config.dt        # initialize configuration N, T, dt
    steps = int(np.round(T / dt))                   # define number of steps from T, dt.

    # Initial condition given configuration mean, standard deviation, and N
    X = rng.normal(config.init_mean, config.init_std, size=N)

    # Optional: cap initial condition on the right
    if config.right_truncate:
        X = np.minimum(X, config.right_cap)

    times, means, vars_ = [], [], []
    snapshots = {}

    # snapshot indices
    fractions = sorted(set(float(f) for f in config.snapshot_fractions))
    snap_indices = set(int(np.round(min(max(f, 0.0), 1.0) * T / dt)) for f in fractions)

    # parameters for HJB-controlled rate
    kappa = 0.5 * config.diffusion_sigma ** 2          # kappa = sigma^2 / 2
    k = config.gamma / (1.0 + config.gamma)            # k = gamma/(1+gamma)
    scale = config.alpha1 * k                          # alpha'(1) = alpha1*k

    loop = range(steps + 1)
    if config.show_progress and tqdm is not None:
        loop = tqdm(loop, total=steps + 1, desc="Simulating", unit="step")

    for n in loop:
        t = n * dt                             # current time   
        if n in snap_indices:                  # save snapshot if at specified fraction
            snapshots[float(t)] = X.copy()     # store copy of current X
        times.append(t)                        # record time
        means.append(float(np.mean(X)))        # record mean
        vars_.append(float(np.var(X)))         # record variance
        if n == steps:                         # end of simulation
            break

        # 1) Diffusion: Euler–Maruyama
        dB = rng.normal(0.0, np.sqrt(dt), size=N)   # Brownian increments
        X = X + config.diffusion_sigma * dB         # Euler–Maruyama step

        # 2) Poisson jump intensity using HJB-controlled rate
        Ik = compute_I_of_X(X, config)                                                  # compute I_k for all particles
        alpha = config.alpha1 * np.minimum(1.0, (scale * Ik)) **config.gamma    # jump rates as in (3.42) of the paper

        # Diagnostics every ~5 time units
        if n % int(5.0 / dt) == 0: 
            I_median = np.median(Ik)             # crude proxy near the bulk; also check top quantiles
            unsat = np.mean(scale * Ik < 1.0)    # fraction unsaturated
            print(f"t={t:.1f}  q90(alpha)={np.quantile(alpha,0.90):.4f}  mean alpha={alpha.mean():.4f}  I_median={I_median:.3f} unsat_frac={unsat:.6f}") 

        # 3) Tau-leap jumps (at most one jump per particle per dt)
        jump_flags = rng.uniform(0.0, 1.0, size=N) < (alpha * dt)                    # determine which particles jump
        if np.any(jump_flags):                                                       # if any jumps occur
            partners = rng.integers(0, N, size=N)                                    # sample partners uniformly
            same = partners == np.arange(N)                                          # resample self-partners
            if np.any(same):                                                         # if any self-partners
                partners[same] = (partners[same] + 1) % N                            # simple fix: shift by one
            X_jump = X.copy()                                                        # copy current X for jumps
            X_jump[jump_flags] = np.maximum(X[jump_flags], X[partners[jump_flags]])  # adopt-the-max jump
            X = X_jump                                                               # update X after jumps

    # Prepare output dictionary
    out = {
        "times": np.asarray(times),
        "means": np.asarray(means),
        "vars": np.asarray(vars_),
        "snapshots": snapshots,
        "final": X.copy(),
        "config": config,
        "kappa": kappa,
        "k": k,
        "alpha_prime1": config.alpha1 * k,
    }
    return out

# ------------------------ Front computation helpers ---------------------- #

def _intrinsic_front_for_snapshot_exact(X, rho, kappa, alpha_prime1):
    """
    Returns (eta_m, e_ell, ok) with e_ell computed from the exact empirical root:
    J(x) = (1/(rho-kappa)) * (e^{-x}/N) * sum_{X_i >= x} e^{X_i} = 1/alpha'(1).
    """
    N = X.size                                 # number of particles
    if N == 0:
        return np.nan, np.nan, False           # no particles ⇒ no front
    Xs = np.sort(X)                            # sorted particle positions
    eta_m = float(np.median(Xs))               # median front

    L = np.empty_like(Xs)                       # log-sum-exp array
    logS = -np.inf                              # log-sum-exp accumulator
    for j in range(N - 1, -1, -1):              # backward iteration for log-sum-exp
        a, b = logS, Xs[j]
        m = max(a, b)
        logS = m + np.log1p(np.exp(min(a, b) - m))
        L[j] = logS

    log_const = np.log(N) + np.log(rho - kappa) - np.log(alpha_prime1)
    x_star = L - log_const

    lefts = np.concatenate((np.array([-np.inf]), Xs[:-1]))
    valids = (x_star >= lefts) & (x_star < Xs)

    if not np.any(valids):
        j = np.argmax(x_star)
        return eta_m, float(x_star[j]), False

    j = int(np.argmax(valids))
    return eta_m, float(x_star[j]), True


def _learning_front_for_snapshot_exact(X, rho, kappa, alpha1, gamma):
    """
    Exact empirical root for the learning-weighted functional J_w(x):
    J_w(x) = (1/(rho-kappa)) * (e^{-x}/N) * sum_{X_i >= x} e^{X_i} w_i = 1/(alpha1*k).
    """
    N = X.size
    if N == 0:
        return np.nan
    Xs = np.sort(X)
    expXs = np.exp(Xs)

    S = np.cumsum(expXs[::-1])[::-1]
    I_sorted = S / (expXs * N)

    k = gamma / (1.0 + gamma)
    scale = alpha1 * k
    w_sorted = np.minimum(1.0, (scale * I_sorted) ** (1.0 + gamma))

    Lw = np.empty_like(Xs)
    logSw = -np.inf
    logew = Xs + np.log(np.maximum(w_sorted, np.finfo(float).tiny))
    for j in range(N - 1, -1, -1):
        a, b = logSw, logew[j]
        m = max(a, b)
        logSw = m + np.log1p(np.exp(min(a, b) - m))
        Lw[j] = logSw

    target = 1.0 / (alpha1 * k)
    log_const = np.log(N) + np.log(rho - kappa) + np.log(target)

    x_star = Lw - log_const
    lefts = np.concatenate((np.array([-np.inf]), Xs[:-1]))
    valids = (x_star >= lefts) & (x_star < Xs)

    if not np.any(valids):
        j = np.argmax(x_star)
        return float(x_star[j])

    j = int(np.argmax(valids))
    return float(x_star[j])


def check_front_equations(snapshot, x_check, rho, kappa, alpha1, gamma, which="intrinsic"):
    
    Xs = np.sort(snapshot)
    expXs = np.exp(Xs)
    S = np.cumsum(expXs[::-1])[::-1]
    N = len(Xs)
    mids = (Xs[:-1] + Xs[1:]) / 2.0
    if len(mids) == 0:
        return float("nan"), float("nan")
    j = np.searchsorted(mids, x_check, side="left")
    j = min(max(j, 1), len(Xs) - 1)

    A_right = S[j] / N
    J = (1.0 / (rho - kappa)) * (np.exp(-x_check) * A_right)

    k = gamma / (1.0 + gamma)
    scale = alpha1 * k
    I_sorted = S / (expXs * N)
    w_sorted = np.minimum(1.0, (scale * I_sorted) ** (1.0 + gamma))
    Aw_right = np.cumsum((expXs * w_sorted)[::-1])[::-1] / N
    Jw = (1.0 / (rho - kappa)) * (np.exp(-x_check) * Aw_right[j])

    target = 1.0 / (alpha1 * k)
    return (J, target) if which == "intrinsic" else (Jw, target)


def track_fronts_over_time_with_learning(result):
    cfg = result["config"]
    rho = float(cfg.rho)
    kappa = float(result["kappa"])
    alpha1 = float(cfg.alpha1)
    gamma = float(cfg.gamma)
    alpha_prime1 = float(result["alpha_prime1"])

    rows = []
    for t in sorted(result["snapshots"].keys()):
        X = result["snapshots"][t]
        eta_m, e_ell, _ = _intrinsic_front_for_snapshot_exact(X, rho, kappa, alpha_prime1)
        eta_l = _learning_front_for_snapshot_exact(X, rho, kappa, alpha1, gamma)
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


def save_histogram_gif(result, path="sim_histograms.gif", bins=100, fps=2, bin_edges=None, x_limits=None, y_limits=None):
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
    a = int(tail_frac * len(ts))
    X = np.vstack([ts[a:], np.ones(len(ts[a:]))]).T
    m, b = np.linalg.lstsq(X, ys[a:], rcond=None)[0]
    return float(m)

# ------------------------------ Main script ------------------------------ #

if __name__ == "__main__":
    # Lottery-society regime: kappa = 1 (sigma = sqrt(2)), rho > 1, alpha1 < 1, gamma = 1 ⇒ k = 1/2
    cfg = SimConfig(
        N=750_000,
        T=1.0,
        dt=1e-5,
        seed=7,
        diffusion_sigma=float(np.sqrt(2.0)),    # kappa = sigma **2 /2 
        rho=2.0,
        alpha1=0.01,                            # weak learning: alpha1 < kappa (< rho)
        gamma=1,                                # gamma = 1 ⇒ k = 1/2 
        right_truncate=False,
        right_cap=10,
        show_progress=True,
    )

    result = simulate(cfg)
    df_fronts = track_fronts_over_time_with_learning(result)
    df_fronts.to_csv("fronts_timeseries.csv", index=False)

    hist_edges, x_limits, y_limits = compute_histogram_scale(result["snapshots"], bins=223)
    gif_path = save_histogram_gif(
        result,
        path="sim_histograms.gif",
        bins=223,
        bin_edges=hist_edges,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    print(f"Histogram GIF saved to {gif_path}")

    snapshot_times = np.asarray(sorted(result["snapshots"].keys()), dtype=float)
    test_times = [cfg.T * frac for frac in (0.25, 0.5, 0.75)]
    rho = float(cfg.rho)
    kappa = float(result["kappa"])
    alpha1 = float(cfg.alpha1)
    gamma = float(cfg.gamma)
    fronts_times = df_fronts["time"].to_numpy(dtype=float)

    for target_time in test_times:
        if snapshot_times.size == 0:
            break
        snap_idx = int(np.abs(snapshot_times - target_time).argmin())
        t_snap = float(snapshot_times[snap_idx])
        snapshot = result["snapshots"][snapshot_times[snap_idx]]
        front_idx = int(np.abs(fronts_times - t_snap).argmin())
        front_row = df_fronts.iloc[front_idx]

        J_intr, target_intr = check_front_equations(
            snapshot,
            x_check=float(front_row["e_ell"]),
            rho=rho,
            kappa=kappa,
            alpha1=alpha1,
            gamma=gamma,
            which="intrinsic",
        )
        J_learn, target_learn = check_front_equations(
            snapshot,
            x_check=float(front_row["eta_l"]),
            rho=rho,
            kappa=kappa,
            alpha1=alpha1,
            gamma=gamma,
            which="learning",
        )

        if not np.isnan(J_intr):
            print(
                f"Front check t≈{t_snap:.2f}: intrinsic J={J_intr:.4f} (target={target_intr:.4f}), "
                f"learning Jw={J_learn:.4f} (target={target_learn:.4f})"
            )
        else:
            print(f"Front check t≈{t_snap:.2f}: insufficient particles for midpoints")

    # Plot fronts
    plt.figure(figsize=(8,4.5))
    plt.plot(df_fronts.time, df_fronts.eta_m, label="eta_m (median)")
    plt.plot(df_fronts.time, df_fronts.e_ell, label="e_ell (intrinsic)")
    plt.plot(df_fronts.time, df_fronts.eta_l, label="eta_ell (learning)")
    plt.legend(); plt.xlabel("time"); plt.ylabel("position"); plt.title("Fronts")
    plt.tight_layout(); plt.show()

    # Slope estimates (late time)
    times_np = np.asarray(df_fronts.time.to_numpy(), dtype=float)
    cm = _fit_slope(times_np, np.asarray(df_fronts.eta_m.to_numpy(), dtype=float))
    cl = _fit_slope(times_np, np.asarray(df_fronts.eta_l.to_numpy(), dtype=float))
    ce = _fit_slope(times_np, np.asarray(df_fronts.e_ell.to_numpy(), dtype=float))
    print(f"Estimated late-time speeds: c_m≈{cm:.3f}, c_ell≈{cl:.3f}, c_intr≈{ce:.3f}")

    # Gap plots
    plt.figure(figsize=(8,4))
    plt.plot(df_fronts.time, df_fronts.eta_l - df_fronts.eta_m)
    plt.xlabel("time"); plt.ylabel("eta_ell - eta_m"); plt.title("Learning - Median gap")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(df_fronts.time, df_fronts.e_ell - df_fronts.eta_l)
    plt.xlabel("time"); plt.ylabel("e_ell - eta_ell"); plt.title("Intrinsic - Learning gap")
    plt.tight_layout(); plt.show()


