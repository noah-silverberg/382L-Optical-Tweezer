import numpy as np
import matplotlib.pyplot as plt

####################################
### RADIUS ADJUSTED-VERSION ########
####################################


# ----------------------------
# Simulation Parameters
# ----------------------------
np.random.seed(47)  # For reproducibility

dt = 1 / 57  # Time step in seconds
M = 100  # Number of particles in the ensemble

# Trajectory length parameters
min_steps = 100  # Minimum number of steps for a trajectory
max_steps = 100  # Maximum number of steps for a trajectory

# True parameters (microns, micron²/s)
v_true = 1.0  # Common drift velocity (microns/s)
D0 = 2.0  # Underlying diffusion constant (micron²/s)
x0 = 0.0  # Initial position (microns)

# ----------------------------
# Introduce Particle-to-Particle Variability in Radius
# ----------------------------
# For example, let the particle radii be uniformly distributed between 0.9 and 1.1.
radii = np.random.uniform(low=0.9, high=1.1, size=M)

# ----------------------------
# Simulate Ensemble of Trajectories with Variable Lengths and Different Radii
# ----------------------------
# We will store each particle's trajectory and its corresponding time vector.
trajectories = []  # List to hold each particle's positions
time_vectors = []  # List to hold the corresponding time vector for each particle
# Also record the particle radii (we already have them in 'radii')

for i in range(M):
    # Randomly choose the number of steps for this particle:
    N_i = np.random.randint(min_steps, max_steps + 1)
    t_i = np.arange(N_i) * dt
    # For this simulation, each particle experiences the same drift v_true.
    # Its individual diffusion constant is D_i = D0 / r_i.
    D_i = D0 / radii[i]
    noise_std_i = np.sqrt(2 * D_i * dt)
    # Generate the diffusive increments and cumulative sum.
    noise = noise_std_i * np.random.randn(N_i)
    diffusive_part = np.cumsum(noise)
    # The trajectory: starting at x0 plus drift and diffusion.
    X_i = x0 + v_true * t_i + diffusive_part
    trajectories.append(X_i)
    time_vectors.append(t_i)

# ----------------------------
# Compute Ensemble-Average Trajectory for Drift Estimation
# ----------------------------
# We first determine the maximum possible number of time steps (which is max_steps)
max_length = max_steps  # maximum possible number of time steps

# Initialize arrays to accumulate the ensemble sum and counts.
ensemble_sum = np.zeros(max_length)
counts = np.zeros(max_length)

# Loop over each particle and add its data (only for time indices it covers).
for traj in trajectories:
    N_i = len(traj)
    ensemble_sum[:N_i] += traj
    counts[:N_i] += 1

# Use only time points where at least 10 particles contribute.
valid_indices = counts >= M / 2
t_valid = np.arange(max_length)[valid_indices] * dt
ensemble_mean_valid = ensemble_sum[valid_indices] / counts[valid_indices]

# Fit a line to the ensemble-average trajectory to estimate the drift.
coeffs = np.polyfit(t_valid, ensemble_mean_valid, 1)
v_est_ensemble, intercept_est_ensemble = coeffs

print(
    f"Ensemble estimated drift: slope = {v_est_ensemble:.4f} micron/s, "
    f"intercept = {intercept_est_ensemble:.4f} micron"
)

# For plotting the drift, we create a full drift curve.
t_common = np.arange(max_length) * dt
drift_global = intercept_est_ensemble + v_est_ensemble * t_common

# ----------------------------
# Drift Correction for Each Particle (Using Global Drift Estimate)
# ----------------------------
trajectories_corr = []  # List to hold drift-corrected trajectories

for i in range(M):
    t_i = time_vectors[i]
    # Evaluate the global drift on this particle's time points.
    drift_i = intercept_est_ensemble + v_est_ensemble * t_i
    # Subtract the drift.
    X_corr_i = trajectories[i] - drift_i
    trajectories_corr.append(X_corr_i)

# ----------------------------
# Compute Ensemble-Averaged Drift-Corrected (and Radius-Scaled) MSD
# ----------------------------
# For each particle, we compute the rescaled squared displacement: r_i * [x_corr]^2.
# Then we average over the ensemble at each time index (only where at least 10 trajectories are available).
msd_rescaled = np.zeros(max_length)
counts_msd = np.zeros(max_length)

for i in range(M):
    traj_corr = trajectories_corr[i]
    N_i = len(traj_corr)
    # Multiply the squared displacement by the particle's radius.
    msd_rescaled[:N_i] += radii[i] * (traj_corr**2)
    counts_msd[:N_i] += 1

# Only compute the ensemble average where at least 10 trajectories contribute.
valid_msd_indices = counts_msd >= 10
t_msd = np.arange(max_length)[valid_msd_indices] * dt
msd_valid = msd_rescaled[valid_msd_indices] / counts_msd[valid_msd_indices]

# ----------------------------
# Linear Fit for Rescaled MSD to Determine D0 and Compute R²
# ----------------------------
# For 1D pure diffusion, theory gives: (rescaled) MSD(t)= 2D0*t.
# We fit a line through the origin using the valid data points.
m = np.sum(t_msd * msd_valid) / np.sum(t_msd**2)
D0_est = m / 2.0  # Estimated universal diffusion constant

# Compute predictions and R².
msd_predicted = m * t_msd
ss_res = np.sum((msd_valid - msd_predicted) ** 2)
ss_tot = np.sum((msd_valid - np.mean(msd_valid)) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"Estimated universal diffusion constant D0 = {D0_est:.4f} micron²/s (ensemble)")
print(f"R² for the rescaled MSD linear fit = {r_squared:.4f}")

# ----------------------------
# Plotting
# ----------------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 14))

# (1) Plot sample original trajectories with the global drift line.
num_samples = min(20, M)
for i in range(num_samples):
    t_i = time_vectors[i]
    traj = trajectories[i]
    axs[0].plot(t_i, traj, color="gray", alpha=0.5)
    # Plot the global drift evaluated on this particle's time vector.
    axs[0].plot(
        t_i, intercept_est_ensemble + v_est_ensemble * t_i, "--", color="red", alpha=0.7
    )
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("x (microns)")
axs[0].set_title("Sample Original Trajectories with Global Drift Lines")

# (2) Plot sample drift-corrected trajectories.
for i in range(num_samples):
    t_i = time_vectors[i]
    axs[1].plot(t_i, trajectories_corr[i], color="gray", alpha=0.5)
# Compute the ensemble average of the drift-corrected trajectories.
ensemble_corr_sum = np.zeros(max_length)
counts_corr = np.zeros(max_length)
for traj_corr in trajectories_corr:
    N_i = len(traj_corr)
    ensemble_corr_sum[:N_i] += traj_corr
    counts_corr[:N_i] += 1
# Only average where at least 10 trajectories are present.
valid_corr_indices = counts_corr >= 10
t_corr = np.arange(max_length)[valid_corr_indices] * dt
ensemble_corr_mean = (
    ensemble_corr_sum[valid_corr_indices] / counts_corr[valid_corr_indices]
)
axs[1].plot(t_corr, ensemble_corr_mean, "b-", lw=2, label="Ensemble Average")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("x_corr (microns)")
axs[1].set_title("Sample Drift-Corrected Trajectories")
axs[1].legend()

# (3) Plot ensemble-averaged radius-scaled MSD and linear fit.
axs[2].scatter(t_msd, msd_valid, s=30, label="Ensemble MSD (rescaled)")
axs[2].plot(
    t_msd,
    msd_predicted,
    "r-",
    lw=2,
    label=(
        f"Linear Fit: slope = {m:.3f}\n"
        f"-> D0 = {D0_est:.3f} micron²/s, R² = {r_squared:.3f}"
    ),
)
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("r * MSD (micron²)")
axs[2].set_title("Ensemble-Averaged Radius-Scaled Drift-Corrected MSD vs. Time")
axs[2].legend()

plt.tight_layout()
plt.show()
