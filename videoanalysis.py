import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output


###############################################
# Helper functions
###############################################
def dist(a, b):
    return np.linalg.norm(a - b)


def circle_from_two_points(p1, p2):
    """
    Given two points (defining a diameter), return the center and radius.
    """
    center = (p1 + p2) / 2.0
    radius = dist(p1, p2) / 2.0
    return center, radius


###############################################
# Step 1: Open video and create output folder & writers
###############################################
video_path = "Videos/diffusion015.mp4"  # video path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: Unable to open video file.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

base_name = os.path.splitext(os.path.basename(video_path))[0]
output_folder = os.path.join("Tracking_Results_V2", base_name)
if os.path.exists(output_folder):
    answer = input(f"Folder '{output_folder}' already exists. Overwrite it? (y/n): ")
    if answer.lower() in ["y", "yes"]:
        shutil.rmtree(output_folder)
        print(f"Existing folder '{output_folder}' has been removed.")
    else:
        print("Exiting without overwriting folder.")
        exit(0)
os.makedirs(output_folder, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_with_path = os.path.join(output_folder, "tracked_with_particle.mp4")
out_blank_path = os.path.join(output_folder, "tracked_blank.mp4")
out_preproc_path = os.path.join(output_folder, "preprocessed_video.mp4")

out_with = cv2.VideoWriter(out_with_path, fourcc, fps, (frame_width, frame_height))
out_blank = cv2.VideoWriter(out_blank_path, fourcc, fps, (frame_width, frame_height))
out_preproc = cv2.VideoWriter(
    out_preproc_path, fourcc, fps, (frame_width, frame_height)
)

###############################################
# Calibration & Tracking Parameters
###############################################

# Containers for tracked particles.
# Each particle is a dictionary with keys:
#   'id': integer ID,
#   'positions': list of (x, y) positions (in pixels),
#   'last_position': last known (x, y),
#   'radius': estimated radius (in pixels),
#   'active': boolean flag indicating whether the particle is still tracked.
tracked_particles = []
next_particle_id = 1

# Search parameters:
search_radius = None  # will be set per particle as 2 * current radius

# Pre-processing parameters
threshold_value = 21  # adjust based on your image contrast
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Gradient Correction Parameter
# Compensates for a bright bottom-right and dark top-left.
alpha = 0.4  # adjust as needed

# Create a window and set mouse callback (for new particle selection)
cv2.namedWindow("Tracking")
new_particle_click = None


def on_mouse(event, x, y, flags, param):
    global new_particle_click
    if event == cv2.EVENT_LBUTTONDOWN:
        new_particle_click = np.array([x, y], dtype=float)


cv2.setMouseCallback("Tracking", on_mouse)

###############################################
# (No initial prompt now; video always starts at t=0)
###############################################
print("Tracking started from t=0.")
print("Press 't' to mark a new particle, 'q' to quit.")

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------
    # Gradient Correction:
    # ---------------------
    rows, cols = frame.shape[:2]
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    xv, yv = np.meshgrid(x, y)
    # Mask ranges from (1 - alpha) in the top-left to (1 + alpha) in the bottom-right.
    mask = (1 - alpha) + (2 * alpha) * ((xv + yv) / 2)
    mask_3 = np.dstack([mask] * 3)
    frame_float = frame.astype(np.float32)
    frame_corrected = frame_float / mask_3
    frame_corrected = np.clip(frame_corrected, 0, 255).astype(np.uint8)

    # For display we continue using the original frame (or you could use frame_corrected)
    frame_disp = frame.copy()
    frame_blank = np.ones_like(frame) * 255  # white background for blank overlay

    # ---------------------
    # Preprocessing: grayscale, blur, threshold, morphology.
    # ---------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blurred, threshold_value, 255, cv2.THRESH_BINARY)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    preproc_vis = cv2.cvtColor(thresh_clean, cv2.COLOR_GRAY2BGR)

    # ---------------------
    # Find contours.
    # ---------------------
    contours, _ = cv2.findContours(
        thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ---------------------
    # Update each tracked (active) particle.
    # ---------------------
    for particle in tracked_particles:
        # Only update particles that are still active.
        if not particle.get("active", True):
            continue

        prev_pos = particle["last_position"]
        best_contour = None
        best_distance = float("inf")
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            candidate = np.array([cx, cy], dtype=float)
            d = dist(candidate, prev_pos)
            if d < best_distance and d < 2 * particle["radius"]:
                best_distance = d
                best_contour = cnt

        # If a matching contour was found, update the particle.
        if best_contour is not None:
            (x_circle, y_circle), detected_radius = cv2.minEnclosingCircle(best_contour)
            new_pos = np.array([x_circle, y_circle], dtype=float)
            particle["positions"].append(new_pos)
            particle["last_position"] = new_pos
            particle["radius"] = 0.8 * particle["radius"] + 0.2 * detected_radius
        else:
            # No contour found: mark the particle as inactive (stop tracking/updating)
            particle["active"] = False

    # ---------------------
    # Draw overlay for active particles.
    # ---------------------
    for particle in tracked_particles:
        if not particle.get("active", True):
            continue
        pos_int = (
            int(round(particle["last_position"][0])),
            int(round(particle["last_position"][1])),
        )
        cv2.circle(frame_disp, pos_int, int(round(particle["radius"])), (0, 255, 0), 2)
        cv2.putText(
            frame_disp,
            f"ID {particle['id']}",
            pos_int,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # ---------------------
    # Show the current frame.
    # ---------------------
    cv2.imshow("Tracking", frame_disp)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("t"):
        print(
            "Press 'y' to add a new particle at this frame, or any other key to continue."
        )
        key2 = cv2.waitKey(0) & 0xFF
        if key2 == ord("y"):
            new_particle_click = None
            print(
                "Click on the new particle's approximate center in the displayed window."
            )
            while new_particle_click is None:
                cv2.waitKey(10)
            approx_center = new_particle_click.copy()
            clear_output(wait=True)
            print(f"Approximate center: {approx_center}")
            # Zoom in for radius selection.
            zoom_size = 25
            x0 = max(0, int(approx_center[0]) - zoom_size)
            x1 = min(frame_width, int(approx_center[0]) + zoom_size)
            y0 = max(0, int(approx_center[1]) - zoom_size)
            y1 = min(frame_height, int(approx_center[1]) + zoom_size)
            zoom_frame = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(zoom_frame, extent=[x0, x1, y1, y0])
            plt.title("Zoomed view: Click two opposite border points")
            plt.xlabel("X (pixels)")
            plt.ylabel("Y (pixels)")
            plt.grid(True)
            plt.plot(
                approx_center[0],
                approx_center[1],
                "rx",
                markersize=10,
                label="Approximate center",
            )
            plt.legend()
            points_2 = plt.ginput(2, timeout=0)
            plt.close()
            p1 = np.array(points_2[0])
            p2 = np.array(points_2[1])
            center_est, radius_est = circle_from_two_points(p1, p2)
            print(
                f"New particle: center = {center_est}, radius = {radius_est:.1f} pixels"
            )
            particle_new = {
                "id": next_particle_id,
                "positions": [center_est],
                "last_position": center_est,
                "radius": radius_est,
                "active": True,  # mark new particle as active
            }
            tracked_particles.append(particle_new)
            next_particle_id += 1
            print(f"Now tracking particle ID {particle_new['id']}.")

    # ---------------------
    # Write outputs to videos.
    # ---------------------
    out_with.write(frame_disp)
    frame_blank = np.ones_like(frame) * 255
    for particle in tracked_particles:
        if not particle.get("active", True):
            continue
        pos = particle["last_position"]
        cv2.circle(
            frame_blank,
            (int(round(pos[0])), int(round(pos[1]))),
            int(round(particle["radius"])),
            (0, 255, 0),
            2,
        )
    out_blank.write(frame_blank)
    out_preproc.write(preproc_vis)

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
out_with.release()
out_blank.release()
out_preproc.release()

###############################################
# Save tracking data to CSV and radii to TXT
###############################################
all_tracking_data = []
for particle in tracked_particles:
    for pos in particle["positions"]:
        all_tracking_data.append(
            {"particle_id": particle["id"], "x": pos[0], "y": pos[1]}
        )
df = pd.DataFrame(all_tracking_data)
csv_path = os.path.join(output_folder, "tracked_positions.csv")
df.to_csv(csv_path, index=False)

radii_data = []
for particle in tracked_particles:
    radii_data.append({"particle_id": particle["id"], "radius": particle["radius"]})
df_radii = pd.DataFrame(radii_data)
radius_txt_path = os.path.join(output_folder, "bead_radius.txt")
df_radii.to_csv(radius_txt_path, index=False)

print("Tracking complete.")
print("Tracked positions saved to:", csv_path)
print("Particle radii saved to:", radius_txt_path)
print("Tracked video saved to:", out_with_path)
print("Blank annotated video saved to:", out_blank_path)
print("Preprocessed video saved to:", out_preproc_path)
