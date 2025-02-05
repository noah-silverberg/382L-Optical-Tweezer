import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== User Input & File Locations =====
video_file = "Videos/diffusiontest5trimmed_02.mp4"
base_name = os.path.splitext(os.path.basename(video_file))[0]
csv_path = os.path.join(base_name, "tracked_positions.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")
radius_txt_path = os.path.join(base_name, "bead_radius.txt")
if not os.path.exists(radius_txt_path):
    raise FileNotFoundError(f"Bead radius file not found at {radius_txt_path}")

# ===== Retrieve Video Frame Rate =====
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise Exception("Error opening video file.")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Frame rate: {fps} frames per second")

# ===== Calibration =====
# 119.88 pixels = 10 microns  =>  conversion factor in µm per pixel
conversion_factor = 10 / 119.88  # µm per pixel

# ===== Load Tracking Data =====
df = pd.read_csv(csv_path)
if "x" not in df.columns or "y" not in df.columns:
    raise Exception("CSV file must contain both 'x' and 'y' columns.")
x = df["x"].values  # in pixels
y = df["y"].values  # in pixels

# Create a time vector in seconds.
n_frames = len(x)
t = np.arange(n_frames) / fps

# ====================  Process the X Direction ====================

# Estimate drift in x by fitting a line: x(t) ≈ x0 + v_x*t.
v_x, intercept_x = np.polyfit(t, x, 1)
x_drift = x[0] + v_x * t
x_corr = x - x_drift  # drift-corrected x displacement (in pixels)

# Compute the one-dimensional MSD for x (in pixels^2),
# then convert to µm^2.
msd_x_pixels = x_corr**2
msd_x = msd_x_pixels * (conversion_factor**2)  # in µm²

# Fit a line through the origin (forced through 0) to msd_x vs. t.
# For 1D diffusion: MSD_x(t) = 2 D_x t  =>  slope = 2 D_x.
slope_x = np.sum(t * msd_x) / np.sum(t**2)
D_x = slope_x / 2.0  # in µm²/s

# Compute goodness-of-fit for the x–data.
fitted_x = slope_x * t
residuals_x = msd_x - fitted_x
rmse_x = np.sqrt(np.mean(residuals_x**2))
ss_res_x = np.sum(residuals_x**2)
ss_tot_x = np.sum((msd_x - np.mean(msd_x)) ** 2)
R2_x = 1 - ss_res_x / ss_tot_x

print(f"\nX-Direction:")
print(f"Estimated drift velocity v_x = {v_x:.4f} pixels/s")
print(f"Estimated diffusion constant D_x = {D_x:.4f} µm²/s")
print(f"R² (x) = {R2_x:.4f}, RMSE (x) = {rmse_x:.4f} µm²")

# ====================  Process the Y Direction ====================

# Estimate drift in y by fitting a line: y(t) ≈ y0 + v_y*t.
v_y, intercept_y = np.polyfit(t, y, 1)
y_drift = y[0] + v_y * t
y_corr = y - y_drift  # drift-corrected y displacement (in pixels)

# Compute the one-dimensional MSD for y (in pixels^2),
# then convert to µm^2.
msd_y_pixels = y_corr**2
msd_y = msd_y_pixels * (conversion_factor**2)  # in µm²

# Fit a line through the origin (forced through 0) to msd_y vs. t.
# For 1D diffusion: MSD_y(t) = 2 D_y t  =>  slope = 2 D_y.
slope_y = np.sum(t * msd_y) / np.sum(t**2)
D_y = slope_y / 2.0  # in µm²/s

# Compute goodness-of-fit for the y–data.
fitted_y = slope_y * t
residuals_y = msd_y - fitted_y
rmse_y = np.sqrt(np.mean(residuals_y**2))
ss_res_y = np.sum(residuals_y**2)
ss_tot_y = np.sum((msd_y - np.mean(msd_y)) ** 2)
R2_y = 1 - ss_res_y / ss_tot_y

print(f"\nY-Direction:")
print(f"Estimated drift velocity v_y = {v_y:.4f} pixels/s")
print(f"Estimated diffusion constant D_y = {D_y:.4f} µm²/s")
print(f"R² (y) = {R2_y:.4f}, RMSE (y) = {rmse_y:.4f} µm²")

# ====================  Plotting ====================

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot for the X-direction.
axs[0].scatter(t, msd_x, s=10, label="Drift-corrected MSD_x (µm²)")
axs[0].plot(t, fitted_x, color="red", label="Linear fit through 0")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("MSD_x (µm²)")
axs[0].set_title("Drift-Corrected MSD in X")
axs[0].legend()
axs[0].text(
    0.05,
    0.95,
    f"D_x = {D_x:.4f} µm²/s\nR² = {R2_x:.4f}\nRMSE = {rmse_x:.4f} µm²",
    transform=axs[0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# Plot for the Y-direction.
axs[1].scatter(t, msd_y, s=10, label="Drift-corrected MSD_y (µm²)")
axs[1].plot(t, fitted_y, color="red", label="Linear fit through 0")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("MSD_y (µm²)")
axs[1].set_title("Drift-Corrected MSD in Y")
axs[1].legend()
axs[1].text(
    0.05,
    0.95,
    f"D_y = {D_y:.4f} µm²/s\nR² = {R2_y:.4f}\nRMSE = {rmse_y:.4f} µm²",
    transform=axs[1].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()

# ====================  Boltzmann Constant Calculation ====================
# Load the bead radius (in pixels) from file.
with open(radius_txt_path, "r") as f:
    bead_radius_pixels = float(f.read().strip())

# Convert bead radius from pixels to meters:
# (pixels → µm using conversion_factor, then µm → m).
bead_radius_m = bead_radius_pixels * conversion_factor * 1e-6

# Convert the diffusion constants from µm²/s to m²/s.
D_x_m2 = D_x * 1e-12
D_y_m2 = D_y * 1e-12

# Temperature and viscosity.
T_Celsius = 24.17
T_Kelvin = T_Celsius + 273.15
eta = 0.9107e-3  # 0.9107 mPa·s = 0.9107e-3 Pa·s

# Estimate k_B from the x and y diffusion constants using Einstein's relation:
kb_x_est = 6 * np.pi * eta * bead_radius_m * D_x_m2 / T_Kelvin
kb_y_est = 6 * np.pi * eta * bead_radius_m * D_y_m2 / T_Kelvin
kb_actual = 1.380649e-23  # J/K (accepted value)

print("\nBoltzmann Constant Calculation:")
print(f"Bead radius: {bead_radius_pixels:.2f} pixels -> {bead_radius_m:.2e} m")
print(f"From X: D_x = {D_x_m2:.2e} m²/s => k_B_x = {kb_x_est:.2e} J/K")
print(f"From Y: D_y = {D_y_m2:.2e} m²/s => k_B_y = {kb_y_est:.2e} J/K")
error_percent_kb_x = 100 * abs(kb_x_est - kb_actual) / kb_actual
error_percent_kb_y = 100 * abs(kb_y_est - kb_actual) / kb_actual
print(f"Percent error in k_B (x): {error_percent_kb_x:.2f}%")
print(f"Percent error in k_B (y): {error_percent_kb_y:.2f}%")

# ====================  Theoretical Diffusion Constant Calculation ====================
# Using the accepted value of k_B, compute the theoretical D.
D_theory = kb_actual * T_Kelvin / (6 * np.pi * eta * bead_radius_m)
print(
    "\nTheoretical Diffusion Constant Calculation (should be same for each direction):"
)
print(f"Theoretical D = {D_theory:.2e} m²/s")
print(f"Measured D_x = {D_x_m2:.2e} m²/s")
print(f"Measured D_y = {D_y_m2:.2e} m²/s")
