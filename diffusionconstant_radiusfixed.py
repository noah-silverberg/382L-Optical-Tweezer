#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#############################################
### ENSEMBLE AVERAGE (NORMAL) DIFFUSION SCRIPT ###
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
# Hard-code a bead radius (in µm); since it's fixed, we do not adjust the MSD by r.
hard_coded_radius_um = 1

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

    # Universal diffusion parameter (in µm²/s) for simulation.
    D_sim = 0.239  # (Note: this is used only to set the noise level)

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
particle_time = []  # time arrays (in seconds)
particle_raw_x = []  # x arrays (in µm)
particle_raw_y = []  # y arrays (in µm)
particle_radii = []  # bead radii (in µm)
particle_lengths = []  # number of frames per particle

#############################################
### LOAD OR SIMULATE THE DATA              ###
#############################################
if use_simulated:
    print("Using simulated data.")
    dt = 1.0 / simulated_fps
    for i in range(simulated_particles):
        n_frames = np.random.randint(min_steps_sim, max_steps_sim + 1)
        t = np.arange(n_frames) * dt

        # For simulation, we could vary the bead radius but then hard-code it below:
        bead_radius_pixels = np.random.uniform(
            bead_radius_pixels_min, bead_radius_pixels_max
        )
        bead_radius_um = bead_radius_pixels * conversion_factor

        # Diffusion coefficient per particle (if one were to scale, but here we only use it for noise level)
        D_i = D_sim / bead_radius_um
        D_i_pixels = D_i / (conversion_factor**2)
        noise_std = np.sqrt(2 * D_i_pixels * dt)

        noise_x = noise_std * np.random.randn(n_frames)
        noise_y = noise_std * np.random.randn(n_frames)
        x = np.cumsum(noise_x) + v_x_sim * t
        y = np.cumsum(noise_y) + v_y_sim * t

        x_um = x * conversion_factor
        y_um = y * conversion_factor

        # Subtract initial position to set trajectories to start at zero.
        x_um = x_um - x_um[0]
        y_um = y_um - y_um[0]

        particle_time.append(t)
        particle_raw_x.append(x_um)
        particle_raw_y.append(y_um)
        particle_radii.append(
            bead_radius_um
        )  # will be replaced by hard-coded value below
        particle_lengths.append(n_frames)
else:
    print("Using real data from folder.")
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
            not os.path.exists(csv_path)
            or not os.path.exists(radius_txt_path)
            or not os.path.exists(video_path)
        ):
            print(f"Missing file in {folder}; skipping.")
            continue
        try:
            df_radius = pd.read_csv(radius_txt_path)
        except Exception as e:
            print(f"Error reading {radius_txt_path}: {e}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        required_columns = ["particle_id", "x", "y"]
        if not all(col in df.columns for col in required_columns):
            print(
                f"CSV file {csv_path} must contain {required_columns}; skipping folder {folder}."
            )
            continue
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
        grouped = df.groupby("particle_id")
        for particle_id, group in grouped:
            group = group.sort_index()
            x = group["x"].values
            y = group["y"].values
            n_frames = len(x)
            if n_frames < 25:
                continue
            t = np.arange(n_frames) / fps
            x_um = x * conversion_factor
            y_um = y * conversion_factor
            x_um = x_um - x_um[0]
            y_um = y_um - y_um[0]
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
### FORCE HARD-CODED RADIUS
#############################################
# Since the radius is hard-coded, override any measured value:
particle_radii = [hard_coded_radius_um for _ in particle_radii]

#############################################
### CHUNKING INTO 2-SECOND SEGMENTS & INTERPOLATION
#############################################
if use_simulated:
    fps_current = simulated_fps
else:
    fps_current = fps
chunk_size = int(round(2 * fps_current))
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
        t_chunk = t_arr[start:end].copy()
        x_chunk = x_arr[start:end].copy()
        y_chunk = y_arr[start:end].copy()
        t_chunk = t_chunk - t_chunk[0]
        x_chunk = x_chunk - x_chunk[0]
        y_chunk = y_chunk - y_chunk[0]
        if t_chunk[-1] < min_duration:
            continue
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

# Compute global drift by fitting a line to the ensemble-averaged trajectories.
(global_v_x, global_intercept_x), cov_x = np.polyfit(
    t_valid_x, ensemble_avg_x, 1, cov=True
)
(global_v_y, global_intercept_y), cov_y = np.polyfit(
    t_valid_y, ensemble_avg_y, 1, cov=True
)

slope_err_x = np.sqrt(cov_x[0, 0])
slope_err_y = np.sqrt(cov_y[0, 0])
intercept_err_x = np.sqrt(cov_x[1, 1])
intercept_err_y = np.sqrt(cov_y[1, 1])

# Compute R² for the global drift fits.
pred_x = global_intercept_x + global_v_x * t_valid_x
ss_res_x = np.sum((ensemble_avg_x - pred_x) ** 2)
ss_tot_x = np.sum((ensemble_avg_x - np.mean(ensemble_avg_x)) ** 2)
R2_x = 1 - ss_res_x / ss_tot_x

pred_y = global_intercept_y + global_v_y * t_valid_y
ss_res_y = np.sum((ensemble_avg_y - pred_y) ** 2)
ss_tot_y = np.sum((ensemble_avg_y - np.mean(ensemble_avg_y)) ** 2)
R2_y = 1 - ss_res_y / ss_tot_y

print("\n=== Global Drift Parameters ===")
print(
    f"Global drift in X: v_x = {global_v_x:.4f} µm/s, intercept = {global_intercept_x:.4f} µm, R² = {R2_x:.3f}"
)
print(
    f"Global drift in Y: v_y = {global_v_y:.4f} µm/s, intercept = {global_intercept_y:.4f} µm, R² = {R2_y:.3f}"
)
print(f"Standard error in slope (v_x): {slope_err_x:.4f}")
print(f"Standard error in slope (v_y): {slope_err_y:.4f}")
print(f"Standard error in intercept (v_x): {intercept_err_x:.4f}")
print(f"Standard error in intercept (v_y): {intercept_err_y:.4f}")

#############################################
### DRIFT CORRECTION (NO RADIUS SCALING)
#############################################
# For each trajectory, subtract the global drift.
particle_driftcorr_x = []
particle_driftcorr_y = []
for t_arr, x_arr, y_arr in zip(particle_time, particle_raw_x, particle_raw_y):
    drift_x = global_intercept_x + global_v_x * t_arr
    drift_y = global_intercept_y + global_v_y * t_arr
    x_corr = x_arr - drift_x
    y_corr = y_arr - drift_y
    particle_driftcorr_x.append(x_corr)
    particle_driftcorr_y.append(y_corr)


def profile_slope(
    t,
    y,
    m_best,
    b_best,
    sigma=None,
    n_grid=61,
    n_sigma=3,
    out_png="profile.png",
    label="",
):
    """
    Fix the slope m on a grid around m_best, re-fit the intercept b at
    each point, and compute χ².  Return the profiled best m and its 1 σ
    error (Δχ² = 1).

    Parameters
    ----------
    t, y        : 1-D arrays   data points
    m_best      : float        slope from initial fit
    b_best      : float        intercept from initial fit
    sigma       : 1-D array or None   per-point σ (None ⇒ equal weights)
    n_grid      : int          grid points in ±n_sigma·σ_m
    n_sigma     : float        half-width in units of initial σ_m
    out_png     : str          file name for diagnostic plot
    label       : str          text for figure title

    Returns
    -------
    m_prof, m_err   profiled slope and its 1 σ error
    """
    # Rough initial σ_m from covariance or 1 % of slope if unknown
    if sigma is None:
        σm_init = abs(0.01 * m_best)  # crude
    else:
        # analytical formula for σ_m in unweighted LS with equal σ
        Sxx = np.sum((t - t.mean()) ** 2)
        σm_init = np.sqrt(np.mean(sigma**2) / Sxx)

    m_grid = np.linspace(m_best - n_sigma * σm_init, m_best + n_sigma * σm_init, n_grid)

    chi2_vals = []
    b_vals = []

    for m in m_grid:
        # Best b for fixed m (weighted or un-weighted)
        if sigma is None:
            b = np.mean(y - m * t)
            resid = y - (m * t + b)
            chi2 = np.sum(resid**2)
        else:
            w = 1.0 / sigma**2
            b = np.sum(w * (y - m * t)) / np.sum(w)
            resid = y - (m * t + b)
            chi2 = np.sum((resid / sigma) ** 2)
        b_vals.append(b)
        chi2_vals.append(chi2)

    chi2_vals = np.asarray(chi2_vals)
    b_vals = np.asarray(b_vals)

    j_min = np.argmin(chi2_vals)
    chi2min = chi2_vals[j_min]
    m_prof = m_grid[j_min]

    # 1-σ where χ² = χ²_min + 1
    try:
        left = np.interp(chi2min + 1, chi2_vals[:j_min][::-1], m_grid[:j_min][::-1])
        right = np.interp(chi2min + 1, chi2_vals[j_min:], m_grid[j_min:])
        m_err = 0.5 * ((m_prof - left) + (right - m_prof))
    except Exception:
        m_err = np.nan

    # ------------- diagnostic figure -------------
    fig, ax1 = plt.subplots(figsize=(5.2, 4.2))
    ax1.plot(m_grid, chi2_vals, "o-", lw=1)
    ax1.axhline(chi2min + 1, c="r", ls="--")
    ax1.axvline(m_prof, c="g", ls="--")
    ax1.set_xlabel(r"Slope $m$ ($\mu\mathrm{m}^2\,\mathrm{s}^{-1}$)")
    ax1.set_ylabel(r"$\chi^2$")
    ax1.set_title(f"χ² profile – {label}")

    # Secondary axis to show best-fit intercept vs m
    ax2 = ax1.twinx()
    ax2.plot(m_grid, b_vals, "k:", alpha=0.6)
    ax2.set_ylabel(r"best $b$ ($\mu\mathrm{m}^2$)", color="k", alpha=0.6)
    # ax2.tick_params(axis="y", labelcolor="k", colors="k", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    # ---------------------------------------------

    return m_prof, m_err


#############################################
### ENSEMBLE AVERAGING OF MSD (NO RADIUS SCALING) FOR X
#############################################
ensemble_sum_MSD_x = np.zeros(max_length)
ensemble_count_MSD_x = np.zeros(max_length)
for t_arr, x_corr in zip(particle_time, particle_driftcorr_x):
    N = len(t_arr)
    ensemble_sum_MSD_x[:N] += x_corr**2
    ensemble_count_MSD_x[:N] += 1

valid_MSD_x = ensemble_count_MSD_x >= min_particles_thresh
t_valid_MSD_x = t_common[valid_MSD_x]
ensemble_avg_MSD_x = ensemble_sum_MSD_x[valid_MSD_x] / ensemble_count_MSD_x[valid_MSD_x]

# ——— compute point-by-point std and sem for the MSD X data ———
msd_vals_x = [[] for _ in range(max_length)]
for traj_x in particle_driftcorr_x:
    for j, dx in enumerate(traj_x):
        msd_vals_x[j].append(dx * dx)

# array of σ_MSD at each lag
msd_std_x = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in msd_vals_x]
)
# standard error of the mean: σ/√N
sem_x = msd_std_x / np.sqrt(ensemble_count_MSD_x)
sigma_point_x = sem_x[valid_MSD_x]  # only keep the valid lags


# ——— weighted linear fit using scipy.curve_fit ———
def lin(t, m, b):
    return m * t + b


popt_x, pcov_x = curve_fit(
    lin,
    t_valid_MSD_x,
    ensemble_avg_MSD_x,
    sigma=sigma_point_x,
    absolute_sigma=True,
)
slope_x, intercept_x = popt_x
sigma_slope_x, sigma_intercept_x = np.sqrt(np.diag(pcov_x))

D_x = slope_x / 2.0
sigma_D_x = sigma_slope_x / 2.0

prof_m_x, prof_σm_x = profile_slope(
    t_valid_MSD_x,
    ensemble_avg_MSD_x,
    slope_x,
    intercept_x,
    sigma=sigma_point_x,  # ← use the real SEM array
    n_sigma=1,
    n_grid=101,
    out_png="profile_slope_X.png",
    label="X (MSD)",
)
msd_pred_x = slope_x * t_valid_MSD_x
ss_res_x = np.sum((ensemble_avg_MSD_x - msd_pred_x) ** 2)
ss_tot_x = np.sum((ensemble_avg_MSD_x - np.mean(ensemble_avg_MSD_x)) ** 2)
R2_MSD_x = 1 - ss_res_x / ss_tot_x

print("\n=== MSD Fit Error Statistics for X (profiled) ===")
print(f"Slope_x = {prof_m_x:.6f} ± {prof_σm_x:.6f} (µm²/s)")
print(f"D_x     = {D_x:.6f} ± {sigma_D_x:.6f} µm²/s")
print(f"R²      = {R2_MSD_x:.3f}")

#############################################
### ENSEMBLE AVERAGING OF MSD (NO RADIUS SCALING) FOR Y
#############################################
ensemble_sum_MSD_y = np.zeros(max_length)
ensemble_count_MSD_y = np.zeros(max_length)
for t_arr, y_corr in zip(particle_time, particle_driftcorr_y):
    N = len(t_arr)
    ensemble_sum_MSD_y[:N] += y_corr**2
    ensemble_count_MSD_y[:N] += 1

valid_MSD_y = ensemble_count_MSD_y >= min_particles_thresh
t_valid_MSD_y = t_common[valid_MSD_y]
ensemble_avg_MSD_y = ensemble_sum_MSD_y[valid_MSD_y] / ensemble_count_MSD_y[valid_MSD_y]

msd_vals_y = [[] for _ in range(max_length)]
for traj_y in particle_driftcorr_y:
    for j, dy in enumerate(traj_y):
        msd_vals_y[j].append(dy * dy)

msd_std_y = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in msd_vals_y]
)
sem_y = msd_std_y / np.sqrt(ensemble_count_MSD_y)
sigma_point_y = sem_y[valid_MSD_y]

popt_y, pcov_y = curve_fit(
    lin,
    t_valid_MSD_y,
    ensemble_avg_MSD_y,
    sigma=sigma_point_y,
    absolute_sigma=True,
)
slope_y, intercept_y = popt_y
sigma_slope_y, sigma_intercept_y = np.sqrt(np.diag(pcov_y))

D_y = slope_y / 2.0
sigma_D_y = sigma_slope_y / 2.0

prof_m_y, prof_σm_y = profile_slope(
    t_valid_MSD_y,
    ensemble_avg_MSD_y,
    slope_y,
    intercept_y,
    sigma=sigma_point_y,
    n_sigma=1,
    n_grid=101,
    out_png="profile_slope_Y.png",
    label="Y (MSD)",
)


msd_pred_y = slope_y * t_valid_MSD_y
ss_res_y = np.sum((ensemble_avg_MSD_y - msd_pred_y) ** 2)
ss_tot_y = np.sum((ensemble_avg_MSD_y - np.mean(ensemble_avg_MSD_y)) ** 2)
R2_MSD_y = 1 - ss_res_y / ss_tot_y

print("\n=== MSD Fit Error Statistics for Y (profiled) ===")
print(f"Slope_y = {prof_m_y:.6f} ± {prof_σm_y:.6f} (µm²/s)")
print(f"D_y     = {D_y:.6f} ± {sigma_D_y:.6f} µm²/s")
print(f"R²      = {R2_MSD_y:.3f}")

#############################################
### AVERAGE DIFFUSION COEFFICIENT (X & Y)
#############################################
D_avg = (D_x + D_y) / 2.0
sigma_D_avg = np.sqrt(sigma_D_x**2 + sigma_D_y**2) / 2.0

print("\n=== Average MSD Fit Statistics ===")
print(f"Average D = {D_avg:.4f} ± {sigma_D_avg:.4f} µm²/s")

# Convert D from µm²/s to m²/s.
D_avg_SI = D_avg * 1e-12
sigma_D_avg_SI = sigma_D_avg * 1e-12

#############################################
### BOLTZMANN CONSTANT CALCULATION
#############################################
# Using the Einstein relation: D = kB T / (6 π η r)
# Here, T and η are given, and r is the hard-coded radius converted to meters.
T_Celsius = 24.78  # °C
T_Kelvin = T_Celsius + 273.15
eta = 0.89e-3  # Pa·s
r_m = hard_coded_radius_um * 1e-6  # convert µm to m

kB = D_avg_SI * 6 * np.pi * eta * r_m / T_Kelvin
sigma_kB = (6 * np.pi * eta * r_m / T_Kelvin) * sigma_D_avg_SI

print("\n=== Boltzmann Constant Calculation ===")
print(f"Average D (SI units) = {D_avg_SI:.4e} m²/s ± {sigma_D_avg_SI:.4e} m²/s")
print(f"Estimated k_B = {kB:.4e} ± {sigma_kB:.4e} J/K")

kb_actual = 1.380649e-23  # J/K
D_theory = kb_actual * T_Kelvin / (6 * np.pi * eta * r_m)  # in m²/s
print("\nTheoretical Universal Diffusion Constant Calculation:")
print(f"Theoretical D = {D_theory:.4e} m²/s")
print(f"Measured ensemble D (from experiment) = {D_avg_SI:.4e} m²/s")

#############################################
### PRE-COMPUTE PLOTTING STATISTICS (Y-DISPLACEMENT)
#############################################
# Here we compute the ensemble average and STD for the drift-corrected y displacement
# and for the corresponding MSD (y^2) for every time index.
max_length_plot = int(np.max(particle_lengths))
y_corr_values = [[] for _ in range(max_length_plot)]
msd_values = [[] for _ in range(max_length_plot)]
for traj in range(len(particle_time)):
    N_i = len(particle_time[traj])
    for j in range(N_i):
        y_val = particle_driftcorr_y[traj][j]
        y_corr_values[j].append(y_val)
        msd_values[j].append(y_val**2)
y_corr_avg = np.array(
    [np.mean(vals) if len(vals) > 0 else np.nan for vals in y_corr_values]
)
y_corr_std = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in y_corr_values]
)
msd_avg = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in msd_values])
msd_std = np.array(
    [np.std(vals, ddof=1) if len(vals) > 1 else np.nan for vals in msd_values]
)

# Use the same valid indices as in the global drift estimation.
valid_plot = ensemble_count_y >= min_particles_thresh
x_valid = t_common[valid_plot]
y_corr_avg_valid = y_corr_avg[valid_plot]
y_corr_std_valid = y_corr_std[valid_plot]
msd_avg_valid = msd_avg[valid_plot]
msd_std_valid = msd_std[valid_plot]

# Re-use the previously computed slope_y for plotting.
m_y_unscaled = slope_y
msd_fit = m_y_unscaled * x_valid
ss_res = np.sum((msd_avg_valid - msd_fit) ** 2)
ss_tot = np.sum((msd_avg_valid - np.mean(msd_avg_valid)) ** 2)
R2 = 1 - ss_res / ss_tot

#############################################
### PLOTTING: DRIFT-CORRECTED Y AND MSD WITH STD BANDS
#############################################
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(18, 5.1), sharex=True)

# LEFT PLOT: Drift-corrected y displacement with STD band.
axs[0].plot(
    x_valid, y_corr_avg_valid, "r--", lw=2, label="Drift-corrected Ensemble Average"
)
axs[0].fill_between(
    x_valid,
    y_corr_avg_valid - y_corr_std_valid,
    y_corr_avg_valid + y_corr_std_valid,
    color="gray",
    alpha=0.3,
    label="±1 STD",
)
axs[0].set_title(r"Drift-corrected $y$ displacement", fontsize=18)
axs[0].set_xlabel("Time (s)", fontsize=16)
axs[0].set_ylabel(r"$y$ displacement ($\mu$m)", fontsize=16)
axs[0].legend(fontsize=14)
axs[0].tick_params(axis="both", which="major", labelsize=14)
axs[0].text(
    0.05,
    0.75,
    f"Corrected drift velocity: $v_y = {global_v_y:.2f}$ $\mu$m/s",
    transform=axs[0].transAxes,
    fontsize=14,
    color="black",
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)

# RIGHT PLOT: Ensemble-averaged unscaled MSD with STD band and linear fit (Y only).
axs[1].plot(x_valid, msd_avg_valid, "r--", lw=2, label="Ensemble MSD")
axs[1].fill_between(
    x_valid,
    msd_avg_valid - msd_std_valid,
    msd_avg_valid + msd_std_valid,
    color="gray",
    alpha=0.3,
    label="±1 STD",
)
axs[1].plot(x_valid, msd_fit, "b-", lw=2, label=f"Fit: y = {m_y_unscaled:.3f} t")
axs[1].set_title("MSD vs. Time", fontsize=18)
axs[1].set_xlabel("Time (s)", fontsize=16)
axs[1].set_ylabel("MSD ($\mu$m$^2$)", fontsize=16)
axs[1].legend(fontsize=14, loc="upper left")
axs[1].tick_params(axis="both", which="major", labelsize=14)
axs[1].text(
    0.05,
    0.75,
    f"Fit: y = {m_y_unscaled:.3f} t\n$R^2$ = {R2:.3f}",
    transform=axs[1].transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
)

fig.tight_layout()
fig.savefig("ytrajectories.png", dpi=300)
plt.show()
