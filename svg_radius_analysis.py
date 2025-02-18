import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Specify your SVG filename with the annotations.
svg_filename = "microscopescale1.svg"

# Parse the SVG file.
tree = ET.parse(svg_filename)
root = tree.getroot()

# SVG files use XML namespaces, so we need to handle that.
# The default SVG namespace is usually: 'http://www.w3.org/2000/svg'
ns = {"svg": "http://www.w3.org/2000/svg"}

# Find all circle elements.
circles = []
for circle in root.findall(".//svg:circle", ns):
    # Extract the circle attributes. These are strings, so convert to float.
    cx = float(circle.attrib.get("cx", "0"))
    cy = float(circle.attrib.get("cy", "0"))
    r = float(circle.attrib.get("r", "0"))
    circles.append((cx, cy, r))

if not circles:
    print("No circles found in the SVG file.")
else:
    # Create a DataFrame.
    df = pd.DataFrame(circles, columns=["x", "y", "radius"])

    # Compute summary statistics for the radii.
    radii = df["radius"]
    count = radii.count()
    mean = radii.mean()
    median = radii.median()
    std_dev = radii.std()
    min_val = radii.min()
    max_val = radii.max()

    print("Summary Statistics for Annotated Circles' Radii (pixels):")
    print(f"Count: {count}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Minimum: {min_val:.2f}")
    print(f"Maximum: {max_val:.2f}")

    # Plot a histogram of the radii.
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(
        radii,
        bins=10,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Annotated Circles' Radii")

    # Add summary statistics to the plot.
    stats_text = (
        f"Count: {count}\n"
        f"Mean: {mean:.2f}\n"
        f"Median: {median:.2f}\n"
        f"Std: {std_dev:.2f}\n"
        f"Min: {min_val:.2f}\n"
        f"Max: {max_val:.2f}"
    )
    plt.gca().text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    plt.show()


px_to_dist = 58.38 / 10  # 53.38 px/10 µm
print(f"Conversion factor: {px_to_dist:.2f} px/µm")
print(f"Mean radius: {mean:.2f} px")
print(f"Mean radius: {mean / px_to_dist:.2f} µm")
print(f"Standard deviation: {std_dev:.2f} px")
print(f"Standard deviation: {std_dev / px_to_dist:.2f} µm")
