import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################################
### ENSEMBLE AVERAGE RADIUS-ADJUSTED ####
#########################################

##############################
### MODE SELECTION         ###
##############################
# Set to True to generate simulated data,
# or False to process real tracking data from a folder.
use_simulated = False

##############################
### CALIBRATION & CONSTANTS###
##############################
# Calibration: 119.88 pixels = 10 µm  => conversion factor in µm per pixel.
conversion_factor = 10 / 119.88  # µm per pixel

##############################################
### SIMULATION PARAMETERS (if simulated)    ###
##############################################
if use_simulated:
    simulated_particles = 10000  # Number of simulated particles.
    min_steps_sim = 1000  # Minimum frames per particle.
    max_steps_sim = 2000  # Maximum frames per particle.
    simulated_fps = 60  # Frame rate in frames per second.

    # Drift parameters (in pixels per second) for simulation.
    v_x_sim = 1.0
    v_y_sim = 1.0

    # Universal diffusion parameter D0 (in µm³/s).
    # For realistic water conditions at room temperature and typical bead sizes (~0.35 µm),
    D0_sim = 0.239

    # Range for bead radii in pixels.
    bead_radius_pixels_min = 4.0
    bead_radius_pixels_max = 5.0
else:
    # For real data, specify the base folder.
    base_folder = "Tracking_Results"
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Base folder '{base_folder}' not found.")

#############################################
### PREPARE CONTAINERS FOR PARTICLE DATA   ###
#############################################
# For each particle we will store:
#   - its time vector (in seconds)
#   - its raw x and y positions (in µm) [before drift correction]
#   - its bead radius (in µm)
#   - its number of frames.
particle_time = []  # list of time arrays (in seconds)
particle_raw_x = []  # list of x arrays (in µm)
particle_raw_y = []  # list of y arrays (in µm)
particle_radii = []  # bead radius for each particle (in µm)
particle_lengths = []  # number of frames for each particle

#############################################
### LOAD OR SIMULATE THE DATA              ###
#############################################
if use_simulated:
    print("Using simulated data.")
    dt = 1.0 / simulated_fps
    for i in range(simulated_particles):
        # Random number of frames for this particle.
        n_frames = np.random.randint(min_steps_sim, max_steps_sim + 1)
        t = np.arange(n_frames) * dt

        # Randomly assign a bead radius (in pixels) and convert to µm.
        bead_radius_pixels = np.random.uniform(
            bead_radius_pixels_min, bead_radius_pixels_max
        )
        bead_radius_um = bead_radius_pixels * conversion_factor

        # Compute the particle’s diffusion constant in µm²/s:
        #   D_i = D0_sim / (bead_radius_um)
        D_i = D0_sim / bead_radius_um  # (units: µm³/s divided by µm gives µm²/s)

        # To simulate in pixel units, convert D_i (µm²/s) to pixel²/s.
        D_i_pixels = D_i / (conversion_factor**2)
        noise_std = np.sqrt(2 * D_i_pixels * dt)  # noise_std in pixels

        # Simulate positions: cumulative sum of diffusive noise plus drift.
        noise_x = noise_std * np.random.randn(n_frames)
        noise_y = noise_std * np.random.randn(n_frames)
        x = np.cumsum(noise_x) + v_x_sim * t  # positions in pixels
        y = np.cumsum(noise_y) + v_y_sim * t  # positions in pixels

        # Convert positions to µm.
        x_um = x * conversion_factor
        y_um = y * conversion_factor

        x_um = x_um - x_um[0]
        y_um = y_um - y_um[0]

        # Store the simulated data.
        particle_time.append(t)
        particle_raw_x.append(x_um)
        particle_raw_y.append(y_um)
        particle_radii.append(bead_radius_um)
        particle_lengths.append(n_frames)
else:
    print("Using real data from folder.")
    # List all sub-folders in base_folder.
    particle_folders = [
        os.path.join(base_folder, d)
        for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ]
    if len(particle_folders) == 0:
        raise Exception(f"No sub-folders found in '{base_folder}'.")
    for folder in particle_folders:
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

        # Read bead radius (in pixels) and convert to µm.
        with open(radius_txt_path, "r") as f:
            try:
                bead_radius_pixels = float(f.read().strip())
            except Exception as e:
                print(f"Error reading {radius_txt_path}: {e}")
                continue
        bead_radius_um = bead_radius_pixels * conversion_factor

        # Get frame rate from video.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(
                f"Warning: Unable to open video {video_path}; skipping folder {folder}."
            )
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            print(
                f"Warning: Invalid fps ({fps}) in video {video_path}; skipping folder {folder}."
            )
            continue

        # Load tracking data.
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
        if n_frames < 25:
            continue  # skip too-short trajectories.
        t = np.arange(n_frames) / fps

        # Convert positions to µm.
        x_um = x * conversion_factor
        y_um = y * conversion_factor

        x_um = x_um - x_um[0]
        y_um = y_um - y_um[0]

        # Store the raw data.
        particle_time.append(t)
        particle_raw_x.append(x_um)
        particle_raw_y.append(y_um)
        particle_radii.append(bead_radius_um)
        particle_lengths.append(n_frames)

#############################################
### GLOBAL (ENSEMBLE) DRIFT ESTIMATION
#############################################
# Because trajectories have variable lengths, compute the ensemble average
# for each time index (in µm) where data exist.
max_length = max(particle_lengths)
ensemble_sum_x = np.zeros(max_length)
ensemble_count_x = np.zeros(max_length)
ensemble_sum_y = np.zeros(max_length)
ensemble_count_y = np.zeros(max_length)

for t_arr, x_arr, y_arr in zip(particle_time, particle_raw_x, particle_raw_y):
    N = len(t_arr)
    ensemble_sum_x[:N] += x_arr
    ensemble_count_x[:N] += 1
    ensemble_sum_y[:N] += y_arr
    ensemble_count_y[:N] += 1

# minimum number of particles contributing at a time index
min_particles = len(particle_lengths) // 2
valid_indices_x = ensemble_count_x >= min_particles
valid_indices_y = ensemble_count_y >= min_particles

# Use a common time vector. (For simulated data, use simulated_fps; for real, assume fps is common.)
if use_simulated:
    fps_used = simulated_fps
else:
    fps_used = fps  # assumed common among real data

t_common = np.arange(max_length) / fps_used
t_valid_x = t_common[valid_indices_x]
t_valid_y = t_common[valid_indices_y]

ensemble_avg_x = ensemble_sum_x[valid_indices_x] / ensemble_count_x[valid_indices_x]
ensemble_avg_y = ensemble_sum_y[valid_indices_y] / ensemble_count_y[valid_indices_y]

# Perform a linear fit to the ensemble-average positions (in µm) vs. time (in s)
global_v_x, global_intercept_x = np.polyfit(t_valid_x, ensemble_avg_x, 1)
global_v_y, global_intercept_y = np.polyfit(t_valid_y, ensemble_avg_y, 1)

print("\n=== Global Drift Parameters ===")
print(
    f"Global drift in X: v_x = {global_v_x:.4f} µm/s, intercept = {global_intercept_x:.4f} µm"
)
print(
    f"Global drift in Y: v_y = {global_v_y:.4f} µm/s, intercept = {global_intercept_y:.4f} µm"
)

#############################################
### DRIFT CORRECTION AND SCALING PER PARTICLE
#############################################
# Subtract the global drift from each particle’s raw data and compute
# the scaled squared displacement: r * [x - drift]^2.
particle_scaled_MSD_x = []
particle_scaled_MSD_y = []

for t_arr, x_arr, y_arr, r in zip(
    particle_time, particle_raw_x, particle_raw_y, particle_radii
):
    drift_x = global_intercept_x + global_v_x * t_arr
    drift_y = global_intercept_y + global_v_y * t_arr
    x_corr = x_arr - drift_x
    y_corr = y_arr - drift_y
    msd_x_scaled = r * (x_corr**2)  # units: (µm)*(µm²)=µm³
    msd_y_scaled = r * (y_corr**2)
    particle_scaled_MSD_x.append(msd_x_scaled)
    particle_scaled_MSD_y.append(msd_y_scaled)

#############################################
### ENSEMBLE AVERAGING OF SCALED MSD
#############################################
ensemble_sum_scaled_x = np.zeros(max_length)
ensemble_count_scaled_x = np.zeros(max_length)
ensemble_sum_scaled_y = np.zeros(max_length)
ensemble_count_scaled_y = np.zeros(max_length)

for t_arr, msd_x_arr, msd_y_arr in zip(
    particle_time, particle_scaled_MSD_x, particle_scaled_MSD_y
):
    N = len(t_arr)
    ensemble_sum_scaled_x[:N] += msd_x_arr
    ensemble_count_scaled_x[:N] += 1
    ensemble_sum_scaled_y[:N] += msd_y_arr
    ensemble_count_scaled_y[:N] += 1

# Use a threshold for a “good” ensemble average.
min_particles = len(particle_lengths) // 2
valid_scaled_x = ensemble_count_scaled_x >= min_particles
valid_scaled_y = ensemble_count_scaled_y >= min_particles

t_valid_scaled_x = t_common[valid_scaled_x]
t_valid_scaled_y = t_common[valid_scaled_y]

ensemble_avg_scaled_MSD_x = (
    ensemble_sum_scaled_x[valid_scaled_x] / ensemble_count_scaled_x[valid_scaled_x]
)
ensemble_avg_scaled_MSD_y = (
    ensemble_sum_scaled_y[valid_scaled_y] / ensemble_count_scaled_y[valid_scaled_y]
)

#############################################
### LINEAR FIT (FORCED THROUGH ZERO)
#############################################
# The theoretical relation is: r * MSD(t) = 2 D0 t,
# so the slope m from a fit through the origin is m = 2 D0.
m_x = np.sum(t_valid_scaled_x * ensemble_avg_scaled_MSD_x) / np.sum(t_valid_scaled_x**2)
D0_x = m_x / 2.0  # in µm³/s
m_y = np.sum(t_valid_scaled_y * ensemble_avg_scaled_MSD_y) / np.sum(t_valid_scaled_y**2)
D0_y = m_y / 2.0
D0_avg = (D0_x + D0_y) / 2.0

msd_predicted_x = m_x * t_valid_scaled_x
msd_predicted_y = m_y * t_valid_scaled_y

ss_res_x = np.sum((ensemble_avg_scaled_MSD_x - msd_predicted_x) ** 2)
ss_tot_x = np.sum((ensemble_avg_scaled_MSD_x - np.mean(ensemble_avg_scaled_MSD_x)) ** 2)
R2_x = 1 - ss_res_x / ss_tot_x

ss_res_y = np.sum((ensemble_avg_scaled_MSD_y - msd_predicted_y) ** 2)
ss_tot_y = np.sum((ensemble_avg_scaled_MSD_y - np.mean(ensemble_avg_scaled_MSD_y)) ** 2)
R2_y = 1 - ss_res_y / ss_tot_y

print("\n=== Ensemble Analysis (Radius–Adjusted) ===")
print(
    f"X–direction: Estimated universal diffusion parameter D0_x = {D0_x:.4f} µm³/s (R² = {R2_x:.4f})"
)
print(
    f"Y–direction: Estimated universal diffusion parameter D0_y = {D0_y:.4f} µm³/s (R² = {R2_y:.4f})"
)
print(f"Ensemble average D0 = {D0_avg:.4f} µm³/s")

#############################################
### PLOTTING ENSEMBLE RESULTS
#############################################
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(
    t_valid_scaled_x, ensemble_avg_scaled_MSD_x, s=10, label="Ensemble MSD_x (scaled)"
)
axs[0].plot(
    t_valid_scaled_x,
    msd_predicted_x,
    color="red",
    label=f"Fit: slope = {m_x:.3f}\n→ D0_x = {D0_x:.3f}",
)
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("r * MSD_x (µm³)")
axs[0].set_title("Ensemble Radius–Scaled MSD in X")
axs[0].legend()

axs[1].scatter(
    t_valid_scaled_y, ensemble_avg_scaled_MSD_y, s=10, label="Ensemble MSD_y (scaled)"
)
axs[1].plot(
    t_valid_scaled_y,
    msd_predicted_y,
    color="red",
    label=f"Fit: slope = {m_y:.3f}\n→ D0_y = {D0_y:.3f}",
)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("r * MSD_y (µm³)")
axs[1].set_title("Ensemble Radius–Scaled MSD in Y")
axs[1].legend()

plt.tight_layout()
plt.show()

#############################################
### BOLTZMANN CONSTANT CALCULATION
#############################################
# Einstein's relation for a particle of radius r is:
#   D = k_B T/(6 π η r).
# Since we assume D_i = D0/r, then D0 = k_B T/(6 π η).
# Our measured D0_avg is in µm³/s; convert to m³/s:
D0_avg_m3 = D0_avg * 1e-18  # 1 µm³ = 1e-18 m³

T_Celsius = 24.17
T_Kelvin = T_Celsius + 273.15
eta = 0.9107e-3  # Pa·s

kb_est = D0_avg_m3 * 6 * np.pi * eta / T_Kelvin
kb_actual = 1.380649e-23  # J/K

print("\n=== Boltzmann Constant Calculation ===")
print(f"Ensemble average D0 (converted) = {D0_avg_m3:.2e} m³/s")
print(f"Estimated k_B = {kb_est:.2e} J/K")
error_percent = 100 * abs(kb_est - kb_actual) / kb_actual
print(f"Percent error in k_B = {error_percent:.2f}%")

#############################################
### THEORETICAL DIFFUSION CONSTANT
#############################################
D_theory = kb_actual * T_Kelvin / (6 * np.pi * eta)  # in m³/s
print("\nTheoretical Universal Diffusion Constant Calculation:")
print(f"Theoretical D0 = {D_theory:.2e} m³/s")
print(f"Measured ensemble D0 (from experiment) = {D0_avg_m3:.2e} m³/s")
