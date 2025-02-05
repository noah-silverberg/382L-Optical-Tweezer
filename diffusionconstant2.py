import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################################
### ENSEMBLE AVERAGE RADIUS-ADJUSTED ####
#########################################


# ===== User Input & Base Folder =====
# Change this to the folder that contains all particle sub-folders.
base_folder = "Tracking_Results"

if not os.path.exists(base_folder):
    raise FileNotFoundError(f"Base folder '{base_folder}' not found.")

# ===== Calibration =====
# 119.88 pixels = 10 microns  => conversion factor in µm per pixel.
conversion_factor = 10 / 119.88  # µm per pixel

# ===== Prepare Lists for Ensemble Data =====
particle_time = []  # list of time vectors for each particle (in seconds)
particle_scaled_MSD_x = (
    []
)  # list of arrays: for each particle, r * (x_corr)^2 (in µm^3, effectively)
particle_scaled_MSD_y = []  # same for y.
particle_lengths = []  # store length (number of frames) for each particle

# Loop over each subfolder in the base folder.
particle_folders = [
    os.path.join(base_folder, d)
    for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
]

if len(particle_folders) == 0:
    raise Exception(f"No sub-folders found in '{base_folder}'.")

for folder in particle_folders:
    # ----- File Paths for This Particle -----
    radius_txt_path = os.path.join(folder, "bead_radius.txt")
    csv_path = os.path.join(folder, "tracked_positions.csv")
    video_path = os.path.join(folder, "tracked_with_particle.mp4")

    if not os.path.exists(radius_txt_path):
        print(f"Warning: {radius_txt_path} not found; skipping folder {folder}.")
        continue
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found; skipping folder {folder}.")
        continue
    if not os.path.exists(video_path):
        print(f"Warning: {video_path} not found; skipping folder {folder}.")
        continue

    # ----- Read the bead radius (in pixels) and convert to µm -----
    with open(radius_txt_path, "r") as f:
        try:
            bead_radius_pixels = float(f.read().strip())
        except Exception as e:
            print(f"Error reading {radius_txt_path}: {e}")
            continue
    bead_radius_um = bead_radius_pixels * conversion_factor  # in µm

    # ----- Get Frame Rate from Video -----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Unable to open video {video_path}; skipping folder {folder}.")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # If fps is 0 or not found, skip.
    if fps <= 0:
        print(
            f"Warning: Invalid fps ({fps}) in video {video_path}; skipping folder {folder}."
        )
        continue

    # ----- Load Tracking Data -----
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        continue
    if "x" not in df.columns or "y" not in df.columns:
        print(
            f"CSV file {csv_path} must contain 'x' and 'y' columns; skipping folder {folder}."
        )
        continue
    x = df["x"].values  # in pixels
    y = df["y"].values  # in pixels

    n_frames = len(x)
    if n_frames < 2:
        continue  # skip if too short.

    t = np.arange(n_frames) / fps  # time vector in seconds

    # ----- Drift Correction in X -----
    # Fit a line: x(t) ≈ x0 + v_x*t.
    v_x, intercept_x = np.polyfit(t, x, 1)
    x_drift = x[0] + v_x * t
    x_corr = x - x_drift  # drift-corrected displacements in pixels

    # Convert x_corr to µm.
    x_corr_um = x_corr * conversion_factor
    # Compute squared displacement and scale by bead radius (in µm).
    # (This scaling is our adjustment so that r * MSD_x ≈ 2 D0 t.)
    msd_x_scaled = bead_radius_um * (x_corr_um**2)  # units: µm * (µm^2) = µm^3

    # ----- Drift Correction in Y -----
    v_y, intercept_y = np.polyfit(t, y, 1)
    y_drift = y[0] + v_y * t
    y_corr = y - y_drift
    y_corr_um = y_corr * conversion_factor
    msd_y_scaled = bead_radius_um * (y_corr_um**2)

    # ----- Store Data for This Particle -----
    particle_time.append(t)
    particle_scaled_MSD_x.append(msd_x_scaled)
    particle_scaled_MSD_y.append(msd_y_scaled)
    particle_lengths.append(n_frames)

# ===== Ensemble Averaging =====
# Determine the maximum trajectory length over the ensemble.
max_length = max(particle_lengths)

# Prepare arrays to accumulate the sum and counts for each time index.
ensemble_sum_x = np.zeros(max_length)
ensemble_count_x = np.zeros(max_length)
ensemble_sum_y = np.zeros(max_length)
ensemble_count_y = np.zeros(max_length)

# Loop over particles and add data (for each time index available).
for t_arr, msd_x_arr, msd_y_arr in zip(
    particle_time, particle_scaled_MSD_x, particle_scaled_MSD_y
):
    N = len(t_arr)
    ensemble_sum_x[:N] += msd_x_arr
    ensemble_count_x[:N] += 1
    ensemble_sum_y[:N] += msd_y_arr
    ensemble_count_y[:N] += 1

# Define a threshold for the minimum number of particles contributing.
min_particles = 2

valid_indices_x = ensemble_count_x >= min_particles
valid_indices_y = ensemble_count_y >= min_particles

t_common = np.arange(max_length) / fps  # common time vector in seconds

t_valid_x = t_common[valid_indices_x]
t_valid_y = t_common[valid_indices_y]

ensemble_avg_scaled_MSD_x = (
    ensemble_sum_x[valid_indices_x] / ensemble_count_x[valid_indices_x]
)
ensemble_avg_scaled_MSD_y = (
    ensemble_sum_y[valid_indices_y] / ensemble_count_y[valid_indices_y]
)

# ===== Linear Fit Through the Origin =====
# For 1D diffusion, we expect: r * MSD(t) = 2 D0 t  =>  slope m = 2D0.
m_x = np.sum(t_valid_x * ensemble_avg_scaled_MSD_x) / np.sum(t_valid_x**2)
D0_x = (
    m_x / 2.0
)  # in µm²/s if the scaled MSD is in µm^3/s (but note that D0 here is the universal value)
m_y = np.sum(t_valid_y * ensemble_avg_scaled_MSD_y) / np.sum(t_valid_y**2)
D0_y = m_y / 2.0

# Compute predicted MSD curves (for plotting).
msd_predicted_x = m_x * t_valid_x
msd_predicted_y = m_y * t_valid_y

# Compute R² for the fits.
ss_res_x = np.sum((ensemble_avg_scaled_MSD_x - msd_predicted_x) ** 2)
ss_tot_x = np.sum((ensemble_avg_scaled_MSD_x - np.mean(ensemble_avg_scaled_MSD_x)) ** 2)
R2_x = 1 - ss_res_x / ss_tot_x

ss_res_y = np.sum((ensemble_avg_scaled_MSD_y - msd_predicted_y) ** 2)
ss_tot_y = np.sum((ensemble_avg_scaled_MSD_y - np.mean(ensemble_avg_scaled_MSD_y)) ** 2)
R2_y = 1 - ss_res_y / ss_tot_y

print("\n=== Ensemble Analysis (Radius–Adjusted) ===")
print(
    f"X–direction: Estimated universal diffusion parameter D0_x = {D0_x:.4f} µm²/s (R² = {R2_x:.4f})"
)
print(
    f"Y–direction: Estimated universal diffusion parameter D0_y = {D0_y:.4f} µm²/s (R² = {R2_y:.4f})"
)

# Optionally average the two directions:
D0_avg = (D0_x + D0_y) / 2.0
print(f"Ensemble average D0 = {D0_avg:.4f} µm²/s")

# ===== Plotting Ensemble Results =====
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(
    t_valid_x, ensemble_avg_scaled_MSD_x, s=10, label="Ensemble MSD_x (scaled)"
)
axs[0].plot(
    t_valid_x,
    msd_predicted_x,
    color="red",
    label=f"Fit: slope={m_x:.3f}\n→ D0_x={D0_x:.3f}",
)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("r * MSD_x (µm³)")
axs[0].set_title("Ensemble Radius–Scaled MSD in X")
axs[0].legend()

axs[1].scatter(
    t_valid_y, ensemble_avg_scaled_MSD_y, s=10, label="Ensemble MSD_y (scaled)"
)
axs[1].plot(
    t_valid_y,
    msd_predicted_y,
    color="red",
    label=f"Fit: slope={m_y:.3f}\n→ D0_y={D0_y:.3f}",
)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("r * MSD_y (µm³)")
axs[1].set_title("Ensemble Radius–Scaled MSD in Y")
axs[1].legend()

plt.tight_layout()
plt.show()

# ===== Boltzmann Constant Calculation =====
# Recall: For a particle of radius r, Einstein's relation gives:
#   D = k_B T/(6 π η r).
# In our model D_i = D0/r, so the universal parameter is:
#   D0 = k_B T/(6 π η).
#
# Convert D0_avg from µm²/s to m²/s:
D0_avg_m2 = D0_avg * 1e-12  # (1 µm²/s = 1e-12 m²/s)

# Temperature and viscosity.
T_Celsius = 24.17
T_Kelvin = T_Celsius + 273.15
eta = 0.9107e-3  # Pa·s

# Calculate Boltzmann constant (should be independent of r):
kb_est = D0_avg_m2 * 6 * np.pi * eta / T_Kelvin
kb_actual = 1.380649e-23  # J/K

print("\n=== Boltzmann Constant Calculation ===")
print(f"Ensemble average D0 (converted) = {D0_avg_m2:.2e} m²/s")
print(f"Estimated k_B = {kb_est:.2e} J/K")
error_percent = 100 * abs(kb_est - kb_actual) / kb_actual
print(f"Percent error in k_B = {error_percent:.2f}%")

# ===== Theoretical Diffusion Constant =====
D_theory = (
    kb_actual * T_Kelvin / (6 * np.pi * eta * 1)
)  # for a particle of radius 1 m, but note that here D0 = k_BT/(6πη) independent of r.
print("\nTheoretical Universal Diffusion Constant Calculation:")
print(f"Theoretical D0 = {D_theory:.2e} m²/s")
print(f"Measured ensemble D0 (from experiment) = {D0_avg_m2:.2e} m²/s")
