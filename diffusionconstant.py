import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################################
### ENSEMBLE AVERAGE RADIUS-ADJUSTED SCRIPT ###
#############################################

##############################
### MODE SELECTION         ###
##############################
# Set to True to generate simulated data,
# or False to process real tracking data from a folder.
use_simulated = False

##############################
### CALIBRATION & CONSTANTS###
##############################
# Calibration: 119.88 pixels = 10 µm => conversion factor in µm per pixel.
conversion_factor = 10 / 239.82  # µm per pixel
hard_coded_radius_um = 1  # hard-code a radius if desired. Otherwise set to False

#############################################
### SIMULATION PARAMETERS (if simulated)    ###
#############################################
if use_simulated:
    simulated_particles = 10000  # Number of simulated particles.
    min_steps_sim = 1000  # Minimum frames per particle.
    max_steps_sim = 2000  # Maximum frames per particle.
    simulated_fps = 60  # Frame rate in frames per second.

    # Drift parameters (in pixels per second) for simulation.
    v_x_sim = 1.0
    v_y_sim = 1.0

    # Universal diffusion parameter D0 (in µm³/s).
    D0_sim = 0.239

    # Range for bead radii in pixels.
    bead_radius_pixels_min = 4.0
    bead_radius_pixels_max = 5.0
else:
    # For real data, specify the new base folder.
    base_folder = "Tracking_Results_V2"
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
        n_frames = np.random.randint(min_steps_sim, max_steps_sim + 1)
        t = np.arange(n_frames) * dt

        bead_radius_pixels = np.random.uniform(
            bead_radius_pixels_min, bead_radius_pixels_max
        )
        bead_radius_um = bead_radius_pixels * conversion_factor

        D_i = D0_sim / bead_radius_um  # D_i in µm²/s
        D_i_pixels = D_i / (conversion_factor**2)
        noise_std = np.sqrt(2 * D_i_pixels * dt)

        noise_x = noise_std * np.random.randn(n_frames)
        noise_y = noise_std * np.random.randn(n_frames)
        x = np.cumsum(noise_x) + v_x_sim * t  # positions in pixels
        y = np.cumsum(noise_y) + v_y_sim * t

        x_um = x * conversion_factor
        y_um = y * conversion_factor

        # Subtract the particle's initial position (for the whole trajectory)
        x_um = x_um - x_um[0]
        y_um = y_um - y_um[0]

        particle_time.append(t)
        particle_raw_x.append(x_um)
        particle_raw_y.append(y_um)
        particle_radii.append(bead_radius_um)
        particle_lengths.append(n_frames)
else:
    print("Using real data from folder.")
    # In the new data format each sub-folder (tracking session) in base_folder contains:
    # - tracked_positions.csv (with columns "particle_id", "x", "y")
    # - bead_radius.txt (CSV file with columns "particle_id", "radius")
    # - tracked_with_particle.mp4 (to extract fps)
    session_folders = [
        os.path.join(base_folder, d)
        for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ]
    if len(session_folders) == 0:
        raise Exception(f"No sub-folders found in '{base_folder}'.")
    for folder in session_folders:
        csv_path = os.path.join(folder, "tracked_positions.csv")
        radius_txt_path = os.path.join(folder, "bead_radius.txt")
        video_path = os.path.join(folder, "tracked_with_particle.mp4")
        if (
            (not os.path.exists(csv_path))
            or (not os.path.exists(radius_txt_path))
            or (not os.path.exists(video_path))
        ):
            print(f"Missing file in {folder}; skipping.")
            continue

        # Read the bead radii file (contains multiple particles).
        try:
            df_radius = pd.read_csv(radius_txt_path)
        except Exception as e:
            print(f"Error reading {radius_txt_path}: {e}")
            continue

        # Read the positions CSV.
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # Ensure the CSV has the required columns.
        required_columns = ["particle_id", "x", "y"]
        if not all(col in df.columns for col in required_columns):
            print(
                f"CSV file {csv_path} must contain {required_columns}; skipping folder {folder}."
            )
            continue

        # Get fps from the video.
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

        # Group the positions data by particle_id.
        grouped = df.groupby("particle_id")
        for particle_id, group in grouped:
            group = group.sort_index()  # ensure proper order
            x = group["x"].values
            y = group["y"].values
            n_frames = len(x)
            if n_frames < 25:
                continue
            t = np.arange(n_frames) / fps

            x_um = x * conversion_factor
            y_um = y * conversion_factor

            # Subtract the particle's initial position.
            x_um = x_um - x_um[0]
            y_um = y_um - y_um[0]

            # Look up the bead radius for this particle.
            row = df_radius[df_radius["particle_id"] == particle_id]
            if row.empty:
                print(
                    f"Could not find bead radius for particle {particle_id} in folder {folder}; skipping particle."
                )
                continue
            bead_radius_pixels = row["radius"].values[0]
            bead_radius_um = bead_radius_pixels * conversion_factor
            if hard_coded_radius_um:
                bead_radius_um = hard_coded_radius_um

            particle_time.append(t)
            particle_raw_x.append(x_um)
            particle_raw_y.append(y_um)
            particle_radii.append(bead_radius_um)
            particle_lengths.append(n_frames)

#############################################
### CHUNKING INTO 2-SECOND SEGMENTS & INTERPOLATION
#############################################
# Use the actual fps to determine the number of frames in 2 seconds.
if use_simulated:
    fps_current = simulated_fps
else:
    fps_current = fps  # assume fps is consistent across sessions
chunk_size = int(round(2 * fps_current))
# Define minimum required duration (in seconds) for a chunk.
min_duration = 0.5  # seconds

chunked_time = []
chunked_raw_x = []
chunked_raw_y = []
chunked_radii = []
chunked_lengths = []

for t_arr, x_arr, y_arr, r, n in zip(
    particle_time, particle_raw_x, particle_raw_y, particle_radii, particle_lengths
):
    num_chunks = int(np.ceil(n / chunk_size))
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        # Extract chunk and reset time and positions relative to the chunk start.
        t_chunk = t_arr[start:end].copy()
        x_chunk = x_arr[start:end].copy()
        y_chunk = y_arr[start:end].copy()
        t_chunk = t_chunk - t_chunk[0]
        x_chunk = x_chunk - x_chunk[0]
        y_chunk = y_chunk - y_chunk[0]
        # Skip chunks that do not span at least min_duration seconds.
        if t_chunk[-1] < min_duration:
            continue
        # Interpolate the data onto a common time grid for the available duration.
        common_t = np.linspace(0, t_chunk[-1], len(t_chunk))
        x_interp = np.interp(common_t, t_chunk, x_chunk)
        y_interp = np.interp(common_t, t_chunk, y_chunk)
        chunked_time.append(common_t)
        chunked_raw_x.append(x_interp)
        chunked_raw_y.append(y_interp)
        chunked_radii.append(r)
        chunked_lengths.append(len(common_t))

# Replace original data with chunked data.
particle_time = chunked_time
particle_raw_x = chunked_raw_x
particle_raw_y = chunked_raw_y
particle_radii = chunked_radii
particle_lengths = chunked_lengths

#############################################
### GLOBAL (ENSEMBLE) DRIFT ESTIMATION
#############################################
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

min_particles_thresh = len(particle_lengths) // 2
valid_indices_x = ensemble_count_x >= min_particles_thresh
valid_indices_y = ensemble_count_y >= min_particles_thresh

if use_simulated:
    fps_used = simulated_fps
else:
    fps_used = fps_current
t_common = np.arange(max_length) / fps_used
t_valid_x = t_common[valid_indices_x]
t_valid_y = t_common[valid_indices_y]

ensemble_avg_x = ensemble_sum_x[valid_indices_x] / ensemble_count_x[valid_indices_x]
ensemble_avg_y = ensemble_sum_y[valid_indices_y] / ensemble_count_y[valid_indices_y]

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
particle_scaled_MSD_x = []
particle_scaled_MSD_y = []

for t_arr, x_arr, y_arr, r in zip(
    particle_time, particle_raw_x, particle_raw_y, particle_radii
):
    drift_x = global_intercept_x + global_v_x * t_arr
    drift_y = global_intercept_y + global_v_y * t_arr
    x_corr = x_arr - drift_x
    y_corr = y_arr - drift_y
    msd_x_scaled = r * (x_corr**2)  # units: µm³
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

min_particles_avg = len(particle_lengths) // 2
valid_scaled_x = ensemble_count_scaled_x >= min_particles_avg
valid_scaled_y = ensemble_count_scaled_y >= min_particles_avg

t_valid_scaled_x = t_common[valid_scaled_x]
t_valid_scaled_y = t_common[valid_scaled_y]

ensemble_avg_scaled_MSD_x = (
    ensemble_sum_scaled_x[valid_scaled_x] / ensemble_count_scaled_x[valid_scaled_x]
)
ensemble_avg_scaled_MSD_y = (
    ensemble_sum_scaled_y[valid_scaled_y] / ensemble_count_scaled_y[valid_scaled_y]
)

# Determine last valid time (where count falls below threshold)
valid_mask_x = ensemble_count_scaled_x >= min_particles_avg
if np.any(~valid_mask_x):
    last_valid_time_x = t_common[np.argmax(~valid_mask_x)]
else:
    last_valid_time_x = t_common[-1]
valid_mask_y = ensemble_count_scaled_y >= min_particles_avg
if np.any(~valid_mask_y):
    last_valid_time_y = t_common[np.argmax(~valid_mask_y)]
else:
    last_valid_time_y = t_common[-1]

#############################################
### LINEAR FIT (FORCED THROUGH ZERO)
#############################################
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
### PLOTTING RESULTS FOR X-DATA (TOP 3 PANELS)
#############################################
fig_x, axs_x = plt.subplots(3, 1, figsize=(12, 14))

# (1) Original x trajectories with global drift.
for i in range(min(20, len(particle_time))):
    axs_x[0].plot(particle_time[i], particle_raw_x[i], color="gray", alpha=0.5)
    axs_x[0].plot(
        particle_time[i],
        global_intercept_x + global_v_x * particle_time[i],
        "--",
        color="red",
        alpha=0.7,
    )
axs_x[0].set_xlabel("Time (s)")
axs_x[0].set_ylabel("x (µm)")
axs_x[0].set_title("X: Sample Original Trajectories with Global Drift")

# (2) Drift-corrected x trajectories with ensemble average.
for i in range(min(20, len(particle_time))):
    x_corr_i = particle_raw_x[i] - (global_intercept_x + global_v_x * particle_time[i])
    axs_x[1].plot(particle_time[i], x_corr_i, color="gray", alpha=0.5)
ensemble_corr_sum_x = np.zeros(max_length)
counts_corr_x = np.zeros(max_length)
for t_arr, x_arr in zip(particle_time, particle_raw_x):
    N = len(t_arr)
    ensemble_corr_sum_x[:N] += x_arr - (global_intercept_x + global_v_x * t_arr)
    counts_corr_x[:N] += 1
valid_corr_x = counts_corr_x >= min_particles_avg
t_corr_x = t_common[valid_corr_x]
ensemble_corr_mean_x = ensemble_corr_sum_x[valid_corr_x] / counts_corr_x[valid_corr_x]
axs_x[1].plot(t_corr_x, ensemble_corr_mean_x, "b-", lw=2, label="Ensemble Average")
axs_x[1].set_xlabel("Time (s)")
axs_x[1].set_ylabel("x_corr (µm)")
axs_x[1].set_title("X: Drift-Corrected Trajectories with Ensemble Average")
axs_x[1].legend()

# (3) Ensemble-averaged radius-scaled MSD for x.
axs_x[2].scatter(
    t_valid_scaled_x, ensemble_avg_scaled_MSD_x, s=30, label="Ensemble MSD_x (scaled)"
)
axs_x[2].plot(
    t_valid_scaled_x,
    msd_predicted_x,
    "r-",
    lw=2,
    label=f"Fit: y = {m_x:.3f} x, R² = {R2_x:.3f}",
)
axs_x[2].set_xlabel("Time (s)")
axs_x[2].set_ylabel("r * MSD_x (µm³)")
axs_x[2].set_title("X: Ensemble-Averaged Radius-Scaled MSD vs. Time")
axs_x[2].legend()

fig_x.tight_layout()
fig_x.savefig("ensemble_analysis_x.png", dpi=300)
plt.show()

#############################################
### PLOTTING RESULTS FOR Y-DATA (TOP 3 PANELS)
#############################################
fig_y, axs_y = plt.subplots(3, 1, figsize=(12, 14))

# (1) Original y trajectories with global drift.
num_to_plot = min(20, len(particle_time))
plot_indices = np.random.choice(len(particle_time), size=num_to_plot, replace=False)
for i in plot_indices:
    axs_y[0].plot(particle_time[i], particle_raw_y[i], color="gray", alpha=0.5)
    axs_y[0].plot(
        particle_time[i],
        global_intercept_y + global_v_y * particle_time[i],
        "--",
        color="red",
        alpha=0.7,
    )
axs_y[0].set_xlabel("Time (s)")
axs_y[0].set_ylabel("y (µm)")
axs_y[0].set_title("Y: Sample Original Trajectories with Global Drift")

# (2) Drift-corrected y trajectories with ensemble average.
num_to_plot = min(20, len(particle_time))
plot_indices = np.random.choice(len(particle_time), size=num_to_plot, replace=False)
for i in plot_indices:
    y_corr_i = particle_raw_y[i] - (global_intercept_y + global_v_y * particle_time[i])
    axs_y[1].plot(particle_time[i], y_corr_i, color="gray", alpha=0.5)
ensemble_corr_sum_y = np.zeros(max_length)
counts_corr_y = np.zeros(max_length)
for t_arr, y_arr in zip(particle_time, particle_raw_y):
    N = len(t_arr)
    ensemble_corr_sum_y[:N] += y_arr - (global_intercept_y + global_v_y * t_arr)
    counts_corr_y[:N] += 1
valid_corr_y = counts_corr_y >= min_particles_avg
t_corr_y = t_common[valid_corr_y]
ensemble_corr_mean_y = ensemble_corr_sum_y[valid_corr_y] / counts_corr_y[valid_corr_y]
axs_y[1].plot(t_corr_y, ensemble_corr_mean_y, "b-", lw=2, label="Ensemble Average")
axs_y[1].set_xlabel("Time (s)")
axs_y[1].set_ylabel("y_corr (µm)")
axs_y[1].set_title("Y: Drift-Corrected Trajectories with Ensemble Average")
axs_y[1].legend()

# (3) Ensemble-averaged radius-scaled MSD for y.
axs_y[2].scatter(
    t_valid_scaled_y, ensemble_avg_scaled_MSD_y, s=30, label="Ensemble MSD_y (scaled)"
)
axs_y[2].plot(
    t_valid_scaled_y,
    msd_predicted_y,
    "r-",
    lw=2,
    label=f"Fit: y = {m_y:.3f} x, R² = {R2_y:.3f}",
)
axs_y[2].set_xlabel("Time (s)")
axs_y[2].set_ylabel("r * MSD_y (µm³)")
axs_y[2].set_title("Y: Ensemble-Averaged Radius-Scaled MSD vs. Time")
axs_y[2].legend()

fig_y.tight_layout()
fig_y.savefig("ensemble_analysis_y.png", dpi=300)
plt.show()

#############################################
### NEW PLOTS: BOTTOM PANEL (Contributing Chunks) & RADIUS DISTRIBUTION
#############################################
# For X-data:
fig_x_bottom, axs_x_bottom = plt.subplots(2, 1, figsize=(12, 10))
axs_x_bottom[0].plot(t_common, ensemble_count_scaled_x, "o-")
axs_x_bottom[0].axhline(
    y=min_particles_avg,
    color="red",
    linestyle="--",
    label=f"Threshold = {min_particles_avg}",
)
axs_x_bottom[0].set_xlabel("Time (s)")
axs_x_bottom[0].set_ylabel("Number of Contributing Chunks")
axs_x_bottom[0].set_title("X: Contributing Chunks vs. Time")
axs_x_bottom[0].legend()

axs_x_bottom[1].hist(
    np.array(particle_radii), bins=20, color="skyblue", edgecolor="black"
)
axs_x_bottom[1].set_xlabel("Bead Radius (µm)")
axs_x_bottom[1].set_ylabel("Frequency")
axs_x_bottom[1].set_title("Distribution of Bead Radii")
fig_x_bottom.tight_layout()
fig_x_bottom.savefig("bottom_panel_and_radii_x.png", dpi=300)
plt.show()

# For Y-data:
fig_y_bottom, axs_y_bottom = plt.subplots(2, 1, figsize=(12, 10))
axs_y_bottom[0].plot(t_common, ensemble_count_scaled_y, "o-")
axs_y_bottom[0].axhline(
    y=min_particles_avg,
    color="red",
    linestyle="--",
    label=f"Threshold = {min_particles_avg}",
)
axs_y_bottom[0].set_xlabel("Time (s)")
axs_y_bottom[0].set_ylabel("Number of Contributing Chunks")
axs_y_bottom[0].set_title("Y: Contributing Chunks vs. Time")
axs_y_bottom[0].legend()

axs_y_bottom[1].hist(
    np.array(particle_radii), bins=20, color="skyblue", edgecolor="black"
)
axs_y_bottom[1].set_xlabel("Bead Radius (µm)")
axs_y_bottom[1].set_ylabel("Frequency")
axs_y_bottom[1].set_title("Distribution of Bead Radii")
fig_y_bottom.tight_layout()
fig_y_bottom.savefig("bottom_panel_and_radii_y.png", dpi=300)
plt.show()

#############################################
### BOLTZMANN CONSTANT CALCULATION
#############################################
D0_avg_m3 = D0_avg * 1e-18  # 1 µm³ = 1e-18 m³
T_Celsius = (
    24.78  # Room temperature (assumed to be equal to the fluid & bead temperature)
)
T_Kelvin = T_Celsius + 273.15
eta = 0.89e-3  # Pa·s (found from online source)
kb_est = D0_avg_m3 * 6 * np.pi * eta / T_Kelvin
kb_actual = 1.380649e-23  # J/K

print("\n=== Boltzmann Constant Calculation ===")
print(f"Ensemble average D0 (converted) = {D0_avg_m3:.2e} m³/s")
print(f"Estimated k_B = {kb_est:.2e} J/K")
error_percent = 100 * abs(kb_est - kb_actual) / kb_actual
print(f"Percent error in k_B = {error_percent:.2f}%")

D_theory = kb_actual * T_Kelvin / (6 * np.pi * eta)  # in m³/s
print("\nTheoretical Universal Diffusion Constant Calculation:")
print(f"Theoretical D0 = {D_theory:.2e} m³/s")
print(f"Measured ensemble D0 (from experiment) = {D0_avg_m3:.2e} m³/s")
