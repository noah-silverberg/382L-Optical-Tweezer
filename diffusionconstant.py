import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

(global_v_x, global_intercept_x), cov_x = np.polyfit(
    t_valid_x, ensemble_avg_x, 1, cov=True
)
(global_v_y, global_intercept_y), cov_y = np.polyfit(
    t_valid_y, ensemble_avg_y, 1, cov=True
)

# Standard errors (the square roots of the diagonal of the covariance matrix)
slope_err_x = np.sqrt(cov_x[0, 0])
slope_err_y = np.sqrt(cov_y[0, 0])
intercept_err_x = np.sqrt(cov_x[1, 1])
intercept_err_y = np.sqrt(cov_y[1, 1])

print("\n=== Global Drift Parameters ===")
print(
    f"Global drift in X: v_x = {global_v_x:.4f} µm/s, intercept = {global_intercept_x:.4f} µm"
)
print(
    f"Global drift in Y: v_y = {global_v_y:.4f} µm/s, intercept = {global_intercept_y:.4f} µm"
)
print(f"Standard error in slope (v_x): {slope_err_x:.4f}")
print(f"Standard error in slope (v_y): {slope_err_y:.4f}")
print(f"Standard error in intercept (v_x): {intercept_err_x:.4f}")
print(f"Standard error in intercept (v_y): {intercept_err_y:.4f}")

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


# -------------------------------------------------------------------
#  WEIGHTED LINEAR FIT  (forced through 0)  –  all raw points
# -------------------------------------------------------------------
def lin0(t, m):  # model with zero intercept
    return m * t


# ── 1) per–lag standard–deviation (σlag) ────────────────────────────
msd_lag_vals_x = [[] for _ in range(max_length)]
msd_lag_vals_y = [[] for _ in range(max_length)]

for traj in range(len(particle_time)):
    for j, (mx, my) in enumerate(
        zip(particle_scaled_MSD_x[traj], particle_scaled_MSD_y[traj])
    ):
        msd_lag_vals_x[j].append(mx)
        msd_lag_vals_y[j].append(my)

σ_lag_x = np.array(
    [np.std(v, ddof=1) if len(v) > 1 else np.nan for v in msd_lag_vals_x]
)
σ_lag_y = np.array(
    [np.std(v, ddof=1) if len(v) > 1 else np.nan for v in msd_lag_vals_y]
)

# ── 2) assemble ONE big array for each axis ─────────────────────────
t_all_x, y_all_x, σ_all_x = [], [], []
t_all_y, y_all_y, σ_all_y = [], [], []

for traj, t_vec in enumerate(particle_time):
    msd_x = particle_scaled_MSD_x[traj]
    msd_y = particle_scaled_MSD_y[traj]

    for j, (mx, my) in enumerate(zip(msd_x, msd_y)):
        # keep only time-lags that (i) have enough particles and (ii) finite σ
        if j == 0:  # toss the (t=0, MSD=0) points
            continue
        if valid_scaled_x[j] and np.isfinite(σ_lag_x[j]) and σ_lag_x[j] > 0:
            t_all_x.append(t_vec[j])
            y_all_x.append(mx)
            σ_all_x.append(σ_lag_x[j])
        if valid_scaled_y[j] and np.isfinite(σ_lag_y[j]) and σ_lag_y[j] > 0:
            t_all_y.append(t_vec[j])
            y_all_y.append(my)
            σ_all_y.append(σ_lag_y[j])

t_all_x = np.asarray(t_all_x)
y_all_x = np.asarray(y_all_x)
σ_all_x = np.asarray(σ_all_x)

t_all_y = np.asarray(t_all_y)
y_all_y = np.asarray(y_all_y)
σ_all_y = np.asarray(σ_all_y)

# ── 3) weighted least-squares slope and its error (analytic) ───────
w_x = 1.0 / σ_all_x**2
Sxx_x = np.sum(w_x * t_all_x**2)
Sxy_x = np.sum(w_x * t_all_x * y_all_x)
m_x_opt = Sxy_x / Sxx_x
σ_m_x = np.sqrt(1.0 / Sxx_x)

w_y = 1.0 / σ_all_y**2
Sxx_y = np.sum(w_y * t_all_y**2)
Sxy_y = np.sum(w_y * t_all_y * y_all_y)
m_y_opt = Sxy_y / Sxx_y
σ_m_y = np.sqrt(1.0 / Sxx_y)

D0_x = m_x_opt / 2.0
D0_x_err = σ_m_x / 2.0
D0_y = m_y_opt / 2.0
D0_y_err = σ_m_y / 2.0

# ── 4) χ² / dof using the *same* raw points ────────────────────────
χ2_x = np.sum(((y_all_x - m_x_opt * t_all_x) / σ_all_x) ** 2)
χ2_y = np.sum(((y_all_y - m_y_opt * t_all_y) / σ_all_y) ** 2)
redχ2_x = χ2_x / (len(t_all_x) - 1)
redχ2_y = χ2_y / (len(t_all_y) - 1)

print("\n=== Weighted fit on ALL raw points (radius-scaled MSD) ===")
print(f"D0_x = {D0_x:.4f} ± {D0_x_err:.4f} µm³/s   (χ²/dof = {redχ2_x:.4f})")
print(f"D0_y = {D0_y:.4f} ± {D0_y_err:.4f} µm³/s   (χ²/dof = {redχ2_y:.4f})")

D0_avg = (D0_x + D0_y) / 2.0
D0_avg_err = np.sqrt(D0_x_err**2 + D0_y_err**2) / 2.0
print(f"\nEnsemble average D0 = {D0_avg:.4f} µm³/s")

# Print the ensemble average D = D0 / r (± error propagation)
print(
    f"Ensemble average D0 / r (± error propagation) = {D0_avg / np.mean(particle_radii):.4f} ± {D0_avg_err / np.mean(particle_radii):.4f} µm²/s"
)


# Compute the ensemble average for original y values (using all trajectories).
ensemble_avg_y_original = np.divide(
    ensemble_sum_y,
    ensemble_count_y,
    out=np.full_like(ensemble_sum_y, np.nan),
    where=ensemble_count_y != 0,
)

# Compute the ensemble average for drift–corrected y values (using all trajectories).
ensemble_corr_sum_y = np.zeros(max_length)
counts_corr_y = np.zeros(max_length)
for t_arr, y_arr in zip(particle_time, particle_raw_y):
    N = len(t_arr)
    # Subtract the global drift from each trajectory.
    ensemble_corr_sum_y[:N] += y_arr - (global_intercept_y + global_v_y * t_arr)
    counts_corr_y[:N] += 1
ensemble_avg_y_corr = np.divide(
    ensemble_corr_sum_y,
    counts_corr_y,
    out=np.full_like(ensemble_corr_sum_y, np.nan),
    where=counts_corr_y != 0,
)

# ==============================================================
# Compute STD bands for drift-corrected y displacement.
# ==============================================================
max_length = int(np.max(particle_lengths))  # maximum length of trajectories
y_corr_all = [[] for _ in range(max_length)]
for traj in range(len(particle_time)):
    N_i = len(particle_time[traj])
    for j in range(N_i):
        val = particle_raw_y[traj][j] - (
            global_intercept_y + global_v_y * particle_time[traj][j]
        )
        y_corr_all[j].append(val)
y_corr_avg = np.array(
    [np.mean(vals) if len(vals) > 0 else np.nan for vals in y_corr_all]
)
y_corr_std = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in y_corr_all]
)

# ==============================================================
# Compute STD bands for the unscaled MSD (drift-corrected).
# ==============================================================
msd_all = [[] for _ in range(max_length)]
for traj in range(len(particle_time)):
    N_i = len(particle_time[traj])
    for j in range(N_i):
        val = particle_raw_y[traj][j] - (
            global_intercept_y + global_v_y * particle_time[traj][j]
        )
        msd_all[j].append(val**2)
msd_avg = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in msd_all])
msd_std = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in msd_all]
)

# Use valid indices where enough trajectories contribute.
valid = ensemble_count_y >= min_particles_avg
x_valid = t_common[valid]
y_corr_avg_valid = y_corr_avg[valid]
y_corr_std_valid = y_corr_std[valid] / np.sqrt(ensemble_count_y[valid])
msd_avg_valid = msd_avg[valid]
msd_std_valid = msd_std[valid] / np.sqrt(ensemble_count_y[valid])

# ==============================================================
# Compute STD bands for the unscaled MSD (drift-corrected) for X
# ==============================================================
msd_all_x = [[] for _ in range(max_length)]
for traj in range(len(particle_time)):
    N_i = len(particle_time[traj])
    for j in range(N_i):
        val_x = particle_raw_x[traj][j] - (
            global_intercept_x + global_v_x * particle_time[traj][j]
        )
        msd_all_x[j].append(val_x**2)
msd_avg_x = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in msd_all_x])
msd_std_x = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in msd_all_x]
)

# Use valid indices (analogous to y)
valid_x = ensemble_count_x >= min_particles_avg
x_valid_x = t_common[valid_x]
msd_avg_valid_x = msd_avg_x[valid_x]
msd_std_valid_x = msd_std_x[valid_x] / np.sqrt(ensemble_count_x[valid_x])


# ==============================================================
# Create the 2-subplot figure with STD bands.
# ==============================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(18, 5.1), sharex=True)

# LEFT PLOT: Drift-corrected y displacement with STD band.
axs[0].plot(
    x_valid,
    y_corr_avg_valid,
    color="tab:red",
    linestyle="--",
    lw=2,
    label="Drift-corrected Ensemble Average",
)
axs[0].fill_between(
    x_valid,
    y_corr_avg_valid - y_corr_std_valid,
    y_corr_avg_valid + y_corr_std_valid,
    color="tab:gray",
    alpha=0.3,
    label="±1 STD",
)
axs[0].set_title(r"Drift-corrected $y$ displacement", fontsize=18)
axs[0].set_xlabel("Time (s)", fontsize=16)
axs[0].set_ylabel(r"$y$ ($\mu$m)", fontsize=16)
axs[0].legend(fontsize=14)
axs[0].tick_params(axis="both", which="major", labelsize=14)
# Adjust annotation position to avoid overlap with the legend.
axs[0].text(
    0.05,
    0.75,
    f"Corrected drift velocity: $v_y = {global_v_y:.2f}$ $\mu$m/s",
    transform=axs[0].transAxes,
    fontsize=14,
    color="0.2",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)

# RIGHT PLOT: Ensemble-averaged unscaled MSD with STD band and linear fit.
# RIGHT PLOT: Ensemble-averaged unscaled MSD with STD band and linear fit.
axs[1].plot(
    x_valid, msd_avg_valid, color="tab:red", linestyle="--", lw=2, label="Ensemble MSD"
)
axs[1].fill_between(
    x_valid,
    msd_avg_valid - msd_std_valid,
    msd_avg_valid + msd_std_valid,
    color="tab:gray",
    alpha=0.3,
    label="±1 STD",
)
axs[1].plot(
    x_valid,
    m_y_opt * x_valid,
    color="tab:blue",
    linestyle="-",
    lw=2,
    label=f"Linear fit",
)
fit_text = (
    r"Fit Equation: MSD = "
    + f"${m_y_opt:.3f}"
    + r" \cdot t$"
    + "\n"
    + r"$\chi^2$/DOF = "
    + f"{redχ2_y:.4f}"
)
axs[1].set_title("MSD vs. Time", fontsize=18)
axs[1].set_xlabel("Time (s)", fontsize=16)
axs[1].set_ylabel("$y^2$ ($\mu$m$^2$)", fontsize=16)
axs[1].legend(fontsize=14, loc="upper left")
axs[1].tick_params(axis="both", which="major", labelsize=14)
# Boxed annotation with the linear fit parameters moved lower.
axs[1].text(
    0.05,
    0.75,
    fit_text,
    transform=axs[1].transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
)


fig.tight_layout()
fig.savefig("ytrajectories.png", dpi=300)
plt.show()


#############################################
### COMMENTED OUT: ORIGINAL MULTIPLE GRAPHS
#############################################
"""
#############################################
### PLOTTING RESULTS FOR X-DATA (TOP 3 PANELS)
#############################################
fig_x, axs_x = plt.subplots(3, 1, figsize=(12, 14))
# ... (original plotting code for X-data)
fig_x.tight_layout()
fig_x.savefig("ensemble_analysis_x.png", dpi=300)
plt.show()

#############################################
### PLOTTING RESULTS FOR Y-DATA (TOP 3 PANELS)
#############################################
fig_y, axs_y = plt.subplots(3, 1, figsize=(12, 14))
# ... (original plotting code for Y-data)
fig_y.tight_layout()
fig_y.savefig("ensemble_analysis_y.png", dpi=300)
plt.show()

#############################################
### NEW PLOTS: BOTTOM PANEL (Contributing Chunks) & RADIUS DISTRIBUTION
#############################################
# For X-data:
fig_x_bottom, axs_x_bottom = plt.subplots(2, 1, figsize=(12, 10))
# ... (original bottom panel code for X-data)
fig_x_bottom.tight_layout()
fig_x_bottom.savefig("bottom_panel_and_radii_x.png", dpi=300)
plt.show()

# For Y-data:
fig_y_bottom, axs_y_bottom = plt.subplots(2, 1, figsize=(12, 10))
# ... (original bottom panel code for Y-data)
fig_y_bottom.tight_layout()
fig_y_bottom.savefig("bottom_panel_and_radii_y.png", dpi=300)
plt.show()
"""

#############################################
### BOLTZMANN CONSTANT CALCULATION
#############################################
D0_avg_m3 = D0_avg * 1e-18  # 1 µm³ = 1e-18 m³
D0_avg_m3_err = D0_avg_err * 1e-18  # 1 µm³ = 1e-18 m³
T_Celsius = (
    24.78  # Room temperature (assumed to be equal to the fluid & bead temperature)
)
T_Kelvin = T_Celsius + 273.15
eta = 0.89e-3  # Pa·s (found from online source)

# Estimate k_B and propagate D0 error
factor = 6 * np.pi * eta / T_Kelvin
kb_est = D0_avg_m3 * factor
kb_est_err = D0_avg_m3_err * factor  # propagated error

kb_actual = 1.380649e-23  # J/K

print("\n=== Boltzmann Constant Calculation ===")
print(f"Ensemble average D0 (converted) = {D0_avg_m3:.2e} m³/s ± {D0_avg_m3_err:.2e}")
print(f"Estimated k_B = {kb_est:.3e} J/K ± {kb_est_err:.3e} J/K")
error_percent = 100 * abs(kb_est - kb_actual) / kb_actual
print(f"Percent error in k_B = {error_percent:.2f}%")

D_theory = kb_actual * T_Kelvin / (6 * np.pi * eta)  # in m³/s
print("\nTheoretical Universal Diffusion Constant Calculation:")
print(f"Theoretical D0 = {D_theory:.2e} m³/s")
print(
    f"Measured ensemble D0 (from experiment) = {D0_avg_m3:.2e} m³/s ± {D0_avg_m3_err:.2e}"
)
