import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########################################################
### Radius analysis script (after radius_measure.py) ###
########################################################

# Path to the CSV file with measured radii.
data_file = "measured_radii.csv"

# Check if the file exists.
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found.")
    exit(1)

# Load the data.
df = pd.read_csv(data_file)

# Check if data is available.
if df.empty:
    print("No data found in", data_file)
    exit(1)

# Extract the measured radii (assumed to be in µm).
radii = df["measured_radius_um"]

# Calculate summary statistics.
mean_radius = np.mean(radii)
std_radius = np.std(radii)
median_radius = np.median(radii)
min_radius = np.min(radii)
max_radius = np.max(radii)

print("Summary Statistics:")
print(f"  Mean   : {mean_radius:.2f} µm")
print(f"  Std Dev: {std_radius:.2f} µm")
print(f"  Median : {median_radius:.2f} µm")
print(f"  Min    : {min_radius:.2f} µm")
print(f"  Max    : {max_radius:.2f} µm")

# Create a figure for the distribution.
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(
    radii, bins=int(np.sqrt(len(radii))), color="skyblue", edgecolor="black", alpha=0.8
)

# Add vertical lines for the mean and one standard deviation.
plt.axvline(
    mean_radius,
    color="red",
    linestyle="dashed",
    linewidth=1.5,
    label=f"Mean = {mean_radius:.2f} µm",
)
plt.axvline(
    mean_radius - std_radius,
    color="orange",
    linestyle="dashed",
    linewidth=1.5,
    label=f"-1 Std = {(mean_radius - std_radius):.2f} µm",
)
plt.axvline(
    mean_radius + std_radius,
    color="orange",
    linestyle="dashed",
    linewidth=1.5,
    label=f"+1 Std = {(mean_radius + std_radius):.2f} µm",
)

# Labeling the plot.
plt.xlabel("Measured Radius (µm)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Measured Radii", fontsize=14)
plt.legend(fontsize=12)

# Optionally annotate the plot with summary stats.
plt.text(
    mean_radius,
    max(n) * 0.8,
    f"Mean = {mean_radius:.2f} µm\nStd = {std_radius:.2f} µm",
    color="red",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.7),
)

plt.tight_layout()

# Save the figure.
output_image = "distribution_of_radii.png"
plt.savefig(output_image, dpi=300)
print(f"\nChart saved as {output_image}")

# Show the plot.
plt.show()
