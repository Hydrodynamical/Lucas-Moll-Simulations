import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


# -------------------------
# Parameters and grid
# -------------------------
L = 200.0          # enlarged spatial domain
N = 1000
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Precompute exp(y - z) - 1 matrix for alpha
diff = x[None, :] - x[:, None]  # diff[j,k] = x_k - x_j
expdiff_minus1 = np.exp(diff) - 1.0

# Initial Gaussian distribution
sigma = 1.5
f = np.exp(-x**2 / (2 * sigma**2))
f /= np.sum(f) * dx  # normalize to unit mass

# Time-stepping parameters
dt = 0.001
T_final = 50.0
num_steps = int(T_final / dt)

# Alpha parameters
alpha1 = 0.8
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
# Storage for snapshots & fronts
# -------------------------
snapshot_times = [0, 10.0, 20.0, 30.0, 40.0, 50.0]
snapshot_indices = [int(t / dt) for t in snapshot_times]
snapshots = {0.0: f.copy()}

times = [0.0]
medians = [compute_median(x, f, dx)]
learning_fronts = []

# Initial A and learning front
kernel0 = expdiff_minus1 * (f[None, :] * dx)
upper0 = np.triu(kernel0)
A0 = np.sum(upper0, axis=1)
learning_fronts.append(compute_learning_front(x, A0))

# -------------------------
# Time integration
# -------------------------
for n in trange(1, num_steps + 1, desc="Simulating Boltzmann MFG", unit="step"):
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
    times.append(t)
    medians.append(compute_median(x, f, dx))
    learning_fronts.append(compute_learning_front(x, A))

    # Save snapshots
    if n in snapshot_indices:
        snapshots[round(t, 5)] = f.copy()

times = np.array(times)
medians = np.array(medians)
learning_fronts = np.array(learning_fronts)

# -------------------------
# Fit asymptotic speeds (linear fits for large times)
# -------------------------
t0 = 15.0  # start time for asymptotic regime
mask = times >= t0

coef_med = np.polyfit(times[mask], medians[mask], 1)           # slope, intercept
coef_lf = np.polyfit(times[mask], learning_fronts[mask], 1)

c_med, b_med = coef_med[0], coef_med[1]
c_lf, b_lf = coef_lf[0], coef_lf[1]


# Plot density snapshots
plt.figure(figsize=(8, 5))
for t in sorted(snapshots.keys()):
    plt.plot(x, snapshots[t], label=f"t={t:g}")
plt.title("Boltzmann MFG evolution from Gaussian initial data (larger domain)")
plt.xlabel("x")
plt.ylabel("f(t,x)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot front trajectories with fitted lines over asymptotic window
plt.figure(figsize=(8, 5))
plt.plot(times, medians, label="median front")
plt.plot(times, learning_fronts, label="learning front (A>=1 boundary)")

# fitted lines over [t0, T_final]
t_fit = np.linspace(t0, T_final, 200)
plt.plot(t_fit, c_med * t_fit + b_med, '--', label=f"median fit, c≈{c_med:.3f}")
plt.plot(t_fit, c_lf * t_fit + b_lf, '--', label=f"learning fit, c≈{c_lf:.3f}")

plt.xlabel("time t")
plt.ylabel("front position")
plt.title(f"Fronts and asymptotic speed fits (fit from t≥{t0})")
plt.legend()
plt.tight_layout()
plt.show()
