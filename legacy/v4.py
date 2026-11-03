import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import csv

# -------------------------
# Parameters and grid
# -------------------------
L = 200.0          # enlarged spatial domain
N = 500
x = np.linspace(-int(L/4), L, N)
dx = x[1] - x[0]

# Precompute exp(y - z) - 1 matrix for alpha
diff = x[None, :] - x[:, None]  # diff[j,k] = x_k - x_j
expdiff_minus1 = np.exp(diff) - 1.0

# Initial Gaussian distribution (baseline)
sigma = 1.5
f0 = np.exp(-x**2 / (2 * sigma**2))
f0 /= np.sum(f0) * dx  # normalize to unit mass

# Time-stepping parameters
dt = 0.001
T_final = 50.0
num_steps = int(T_final / dt)

# Alpha parameters (gamma fixed)
gamma = 2.0

# -------------------------
# Helper functions
# -------------------------
def laplacian_neumann(f, dx):
    """Second-order Laplacian with Neumann BCs on a uniform grid."""
    lap = np.zeros_like(f)
    lap[1:-1] = (f[:-2] - 2 * f[1:-1] + f[2:]) / dx**2
    lap[0] = (f[1] - f[0]) / dx**2
    lap[-1] = (f[-2] - f[-1]) / dx**2
    return lap

def compute_median(x, f, dx):
    """Return the median position of the probability density f on grid x."""
    cdf = np.cumsum(f) * dx
    idx = np.searchsorted(cdf, 0.5)
    if idx == 0:
        return x[0]
    if idx >= len(x):
        return x[-1]
    cdf_left = cdf[idx - 1]
    cdf_right = cdf[idx]
    w = (0.5 - cdf_left) / (cdf_right - cdf_left + 1e-16)
    return x[idx - 1] + w * (x[idx] - x[idx - 1])

def compute_learning_front(x, A):
    """
    Learning front as the last point (moving left->right) where A(z,f) >= 1,
    i.e. the rightmost point of the saturated region {A >= 1}.
    If A never reaches 1, return np.nan.
    """
    idxs = np.where(A >= 1.0)[0]
    if len(idxs) == 0:
        return np.nan
    j = idxs[-1]  # rightmost index with A >= 1
    if j == len(x) - 1:
        return x[j]
    A_left = A[j]
    A_right = A[j + 1]
    if A_right == A_left:
        return x[j]
    # interpolate between x[j] (A_left >= 1) and x[j+1] (A_right < 1)
    w = (1.0 - A_left) / (A_right - A_left)
    return x[j] + w * (x[j + 1] - x[j])

# -------------------------
# Single-simulation runner
# -------------------------
def run_simulation(alpha1, t0_fit=15.0, show_progress=False):
    """
    Run the Boltzmann MFG simulation for a given alpha1 (learning_rate)
    and return the asymptotic speeds of the median and learning fronts.

    Returns:
        c_med, c_lf  (slopes from linear fits over t >= t0_fit)
    """
    # copy initial condition
    f = f0.copy()

    times = [0.0]
    medians = [compute_median(x, f, dx)]

    # Initial A and learning front
    kernel0 = expdiff_minus1 * (f[None, :] * dx)
    upper0 = np.triu(kernel0)
    A0 = np.sum(upper0, axis=1)
    learning_fronts = [compute_learning_front(x, A0)]

    iterator = range(1, num_steps + 1)
    if show_progress:
        iterator = trange(1, num_steps + 1,
                          desc=f"Sim alpha={alpha1:.3f}", unit="step")

    for n in iterator:
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
        L_term = f * alpha * tail

        # Diffusion
        f_xx = laplacian_neumann(f, dx)

        # Explicit Euler update
        f = f + dt * (f_xx + G - L_term)

        # Enforce nonnegativity and renormalize
        f = np.maximum(f, 0.0)
        mass = np.sum(f) * dx
        if mass > 0:
            f /= mass

        # time and fronts
        t = n * dt
        times.append(t)
        medians.append(compute_median(x, f, dx))
        learning_fronts.append(compute_learning_front(x, A))

    times = np.array(times)
    medians = np.array(medians)
    learning_fronts = np.array(learning_fronts)

    # -------------------------
    # Fit asymptotic speeds
    # -------------------------
    # median: should always be finite
    mask_med = times >= t0_fit
    coef_med = np.polyfit(times[mask_med], medians[mask_med], 1)
    c_med, b_med = coef_med[0], coef_med[1]

    # learning front: might have NaNs if A never saturates
    mask_lf = (times >= t0_fit) & np.isfinite(learning_fronts)
    if np.sum(mask_lf) >= 2:
        coef_lf = np.polyfit(times[mask_lf], learning_fronts[mask_lf], 1)
        c_lf, b_lf = coef_lf[0], coef_lf[1]
    else:
        c_lf = np.nan  # no reliable learning-front speed

    return c_med, c_lf

# -------------------------
# Parameter sweep over learning_rate (alpha1)
# -------------------------
learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0, 1.01, 1.1, 1.2, 2.0, 2.5, 3.0, 3.5, 4.0]

median_speeds = []
learning_speeds = []

print("Running simulations over learning_rate (alpha1)...")
for alpha1 in tqdm(learning_rates, desc="Alphas"):
    c_med, c_lf = run_simulation(alpha1, t0_fit=15.0, show_progress=False)
    median_speeds.append(c_med)
    learning_speeds.append(c_lf)

median_speeds = np.array(median_speeds)
learning_speeds = np.array(learning_speeds)

# -------------------------
# Save results to CSV
# -------------------------
csv_filename = "front_speeds_vs_alpha.csv"
with open(csv_filename, mode="w", newline="") as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["learning_rate", "c_med", "c_learning_front"])
    for a, c_m, c_l in zip(learning_rates, median_speeds, learning_speeds):
        writer.writerow([a, c_m, c_l])

print(f"Saved speeds to {csv_filename}")

# -------------------------
# Plot speeds vs learning_rate
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, median_speeds, "o-", label="median front speed")
plt.plot(learning_rates, learning_speeds, "s--", label="learning front speed")

plt.axvline(1.0, linestyle=":", label="alpha = 1", linewidth=1)

plt.xlabel("learning_rate (alpha1)")
plt.ylabel("asymptotic front speed c")
plt.title("Front speeds vs learning_rate")
plt.legend()
plt.tight_layout()
plt.show()
