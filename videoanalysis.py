import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from IPython.display import display, clear_output
import time


###############################################
# Helper functions
###############################################
def dist(a, b):
    return np.linalg.norm(a - b)


def circle_from_two_points(p1, p2):
    """
    Given two points (defining a diameter), return the center and radius
    of the circle.
    """
    center = (p1 + p2) / 2.0
    radius = dist(p1, p2) / 2.0
    return center, radius


###############################################
# Step 1: Show the first frame with pixel grid
###############################################
video_path = "Videos/11_02.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: Unable to open video file.")

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    raise Exception("Error: Unable to read the first frame.")

# Convert to RGB for plotting
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
frame_height, frame_width = first_frame.shape[:2]

plt.figure(figsize=(8, 6))
plt.imshow(first_frame_rgb)
plt.title("Step 1: Click the particle's approximate center")
plt.grid(True)
plt.xticks(np.arange(0, frame_width, step=50))
plt.yticks(np.arange(0, frame_height, step=50))
# Wait for single click (the approximate center)
initial_click = plt.ginput(1, timeout=0)[0]
plt.close()
initial_x, initial_y = int(initial_click[0]), int(initial_click[1])
print(f"Initial position (approximate center): x={initial_x}, y={initial_y}")

###############################################
# Step 2: Zoom in and select two points on the particle's border
###############################################
zoom_size = 25  # Half-size of the zoomed window

# Ensure the bounding box is within the image boundaries
x0 = max(0, initial_x - zoom_size)
x1 = min(frame_width, initial_x + zoom_size)
y0 = max(0, initial_y - zoom_size)
y1 = min(frame_height, initial_y + zoom_size)

# Crop the zoomed region (still in full-frame coordinates)
zoom_frame = first_frame_rgb[y0:y1, x0:x1]

# Display the zoomed image with an extent that matches full-frame coordinates.
plt.figure(figsize=(6, 6))
plt.imshow(zoom_frame, extent=[x0, x1, y1, y0])
plt.title(
    "Step 2: Zoomed view\nClick two opposite points on the particle's border (e.g., left and right edges)"
)
plt.xlabel("X coordinate (pixels)")
plt.ylabel("Y coordinate (pixels)")
plt.grid(True)
plt.plot(initial_x, initial_y, "rx", markersize=10, label="Approximate center")
plt.legend()
# Wait for two clicks (the two opposite points along a diameter)
points_2 = plt.ginput(2, timeout=0)
plt.close()

# The points are now in absolute image coordinates
p1 = np.array(points_2[0])
p2 = np.array(points_2[1])
center_est, radius_est = circle_from_two_points(p1, p2)
print(
    f"Estimated center from border clicks: ({center_est[0]:.1f}, {center_est[1]:.1f}), radius: {radius_est:.1f} pixels"
)

# Use these as the initial position and size for tracking.
initial_x, initial_y = int(round(center_est[0])), int(round(center_est[1]))
radius = radius_est  # floating-point radius
expected_area = np.pi * (radius**2)

###############################################
# Step 3: Track the particle in the video
###############################################
# Re-open the video file
cap.release()
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: Unable to reopen video file.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create an output folder named after the video (minus extension)
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_folder = os.path.join("Tracking_Results", base_name)
os.makedirs(output_folder, exist_ok=True)

# Define video output paths
out_with_path = os.path.join(output_folder, "tracked_with_particle.mp4")
out_blank_path = os.path.join(output_folder, "tracked_blank.mp4")
out_preproc_path = os.path.join(output_folder, "preprocessed_video.mp4")

out_with = cv2.VideoWriter(out_with_path, fourcc, fps, (frame_width, frame_height))
out_blank = cv2.VideoWriter(out_blank_path, fourcc, fps, (frame_width, frame_height))
out_preproc = cv2.VideoWriter(
    out_preproc_path, fourcc, fps, (frame_width, frame_height)
)

tracked_positions = []
prev_position = np.array([initial_x, initial_y])
search_radius = 2 * radius  # allowable movement distance; adjust if necessary

# Pre-processing parameters
threshold_value = 20  # adjust based on your image contrast
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare copies for overlays
    frame_with = frame.copy()
    frame_blank = np.ones_like(frame) * 255  # white background for one output

    # Convert frame to grayscale and apply a slight Gaussian blur for noise reduction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to create a binary image
    _, thresh = cv2.threshold(gray_blurred, threshold_value, 255, cv2.THRESH_BINARY)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Convert binary image to BGR for visualization
    preproc_vis = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_distance = float("inf")

    # Loop through all detected contours
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        # Get centroid of the contour
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        candidate_pos = np.array([cx, cy])
        distance = dist(candidate_pos, prev_position)
        area = cv2.contourArea(cnt)

        # Area filter: allow contours whose area is within a certain fraction of the expected area.
        if abs(area - expected_area) > expected_area * 0.5:
            continue

        # Distance filter: choose the contour closest to the previous position
        if distance < best_distance and distance < search_radius:
            best_distance = distance
            best_contour = cnt

    # If no suitable contour is found, we assume the bead is lost and stop processing.
    if best_contour is None:
        print("Lost track of bead. Stopping video processing.")
        break

    # Use the best contour to update position and radius.
    (x_circle, y_circle), detected_radius = cv2.minEnclosingCircle(best_contour)
    tracked_position = np.array([x_circle, y_circle])
    # Smoothly update the particle radius (80% previous, 20% new measurement)
    radius = 0.8 * radius + 0.2 * detected_radius
    expected_area = np.pi * (radius**2)

    tracked_positions.append(tracked_position.tolist())
    prev_position = tracked_position

    # Draw overlays on the outputs
    center_int = (int(round(tracked_position[0])), int(round(tracked_position[1])))
    cv2.circle(frame_with, center_int, int(round(radius)), (0, 255, 0), 2)
    cv2.circle(frame_with, center_int, 2, (0, 0, 255), -1)
    cv2.putText(
        frame_with,
        f"Frame: {frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.circle(preproc_vis, center_int, int(round(radius)), (0, 255, 0), 2)
    cv2.circle(preproc_vis, center_int, 2, (0, 0, 255), -1)
    cv2.putText(
        preproc_vis,
        f"Frame: {frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.circle(frame_blank, center_int, int(round(radius)), (0, 255, 0), 2)
    cv2.circle(frame_blank, center_int, 2, (0, 0, 255), -1)
    cv2.putText(
        frame_blank,
        f"Frame: {frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # Write frames to the output videos
    out_with.write(frame_with)
    out_blank.write(frame_blank)
    out_preproc.write(preproc_vis)

    frame_index += 1

cap.release()
out_with.release()
out_blank.release()
out_preproc.release()

###############################################
# Step 4: Save tracking data to CSV and radius to TXT
###############################################
df = pd.DataFrame(tracked_positions, columns=["x", "y"])
csv_path = os.path.join(output_folder, "tracked_positions.csv")
df.to_csv(csv_path, index=False)

# Save the final estimated radius to a text file.
radius_txt_path = os.path.join(output_folder, "bead_radius.txt")
with open(radius_txt_path, "w") as f:
    f.write(f"{radius:.2f}")

print("Tracking complete.")
print("Videos saved:")
print(f" - {out_with_path}")
print(f" - {out_blank_path}")
print(f" - {out_preproc_path}")
print(f"Tracking data saved to: {csv_path}")
print(f"Bead radius saved to: {radius_txt_path}")
