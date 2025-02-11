import os
import cv2
import numpy as np
import pandas as pd

################################################################################
### Script for getting more accurate radii than the tracking algorithm gives ###
################################################################################

# Conversion factor (use the same as in your tracking script)
conversion_factor = 10 / 239.82  # µm per pixel


# ----- Helper: Extract a square ROI centered on a given coordinate -----
def get_centered_roi(frame, center, half_size):
    """
    Extract a square ROI of size (2*half_size) x (2*half_size) from the frame,
    centered at 'center'. Uses rounding for centering and pads with black if needed.
    """
    x_center = int(round(center[0]))
    y_center = int(round(center[1]))
    h, w = frame.shape[:2]
    left = x_center - half_size
    right = x_center + half_size
    top = y_center - half_size
    bottom = y_center + half_size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    left_clip = max(left, 0)
    top_clip = max(top, 0)
    right_clip = min(right, w)
    bottom_clip = min(bottom, h)

    roi = frame[top_clip:bottom_clip, left_clip:right_clip].copy()
    if pad_top or pad_bottom or pad_left or pad_right:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    return roi


# ----- Helper: Interactive radius measurement -----
def measure_radius(image):
    """
    Displays the provided image in a window.
    The user should click twice:
      1. First on the particle's CENTER.
      2. Then on a point on its EDGE.
    The Euclidean distance (in pixels) is computed and returned.
    Returns the radius in pixels (or None if cancelled).
    """
    clicks = []
    window_name = "Measure (click center then edge; ESC to cancel)"
    clone = image.copy()

    def click_event(event, x, y, flags, param):
        nonlocal clicks, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(clone, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)
    print(
        "In the 'Measure' window, click once on the CENTER of the particle, then on its EDGE."
    )

    while len(clicks) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC cancels measurement
            cv2.destroyWindow(window_name)
            print("Measurement cancelled.")
            return None
    pt_center, pt_edge = clicks[0], clicks[1]
    radius_pixels = np.sqrt(
        (pt_center[0] - pt_edge[0]) ** 2 + (pt_center[1] - pt_edge[1]) ** 2
    )
    cv2.circle(clone, pt_center, int(radius_pixels), (255, 0, 0), 2)
    cv2.imshow(window_name, clone)
    print(f"Measured radius: {radius_pixels:.2f} pixels. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return radius_pixels


# ----- Main Script -----
# Directories (adjust these paths as needed)
tracking_root = "Tracking_Results_V2"  # Contains sub-folders with tracking data
video_root = "Videos"  # Contains video files (e.g., diffusion015.mp4)
output_csv = "measured_radii.csv"  # Output CSV file for measurements

# Load existing measurements (if any) so that we can resume later.
if os.path.exists(output_csv):
    measured_df = pd.read_csv(output_csv)
    print(f"Loaded {len(measured_df)} existing measurements from {output_csv}.")
else:
    measured_df = pd.DataFrame(columns=["folder", "particle_id", "measured_radius_um"])

# Create a set of (folder, particle_id) pairs already processed.
processed_particles = set()
for idx, row in measured_df.iterrows():
    processed_particles.add((row["folder"], row["particle_id"]))

# This list will hold new measurements in this run.
results = []

# Get list of session folders.
session_folders = sorted(
    [
        d
        for d in os.listdir(tracking_root)
        if os.path.isdir(os.path.join(tracking_root, d))
    ]
)

quit_all = False  # flag to quit the entire script

for folder in session_folders:
    session_path = os.path.join(tracking_root, folder)
    pos_file = os.path.join(session_path, "tracked_positions.csv")
    if not os.path.exists(pos_file):
        print(f"{pos_file} not found; skipping folder '{folder}'.")
        continue
    try:
        df_positions = pd.read_csv(pos_file)
    except Exception as e:
        print(f"Error reading {pos_file}: {e}")
        continue

    particle_ids = sorted(df_positions["particle_id"].unique())
    print(f"\nProcessing folder '{folder}' with {len(particle_ids)} particles.")

    video_file = os.path.join(video_root, f"{folder}.mp4")
    if not os.path.exists(video_file):
        print(f"Video file '{video_file}' not found; skipping folder '{folder}'.")
        continue
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video '{video_file}'; skipping folder '{folder}'.")
        continue

    roi_half_size = 100  # ROI will be 200x200 pixels; adjust as needed.

    for pid in particle_ids:
        # Skip if this particle has already been processed.
        if (folder, pid) in processed_particles:
            print(f"Skipping folder '{folder}', particle {pid} (already measured).")
            continue

        particle_rows = df_positions[df_positions["particle_id"] == pid]
        if particle_rows.empty:
            continue
        N = len(particle_rows)
        print(f"\nParticle {pid}: {N} tracked frames available in folder '{folder}'.")
        selected_frame = None

        # Loop over the tracking data for this particle.
        # We assume row i corresponds to video frame i.
        for i in range(N):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print(
                    f"Could not read frame {i} from video; stopping playback for this particle."
                )
                break
            row = particle_rows.iloc[i]
            center = (float(row["x"]), float(row["y"]))
            roi = get_centered_roi(frame, center, roi_half_size)
            cv2.imshow("Zoomed Particle", roi)
            # Playback at 0.25x speed (120 ms per frame).
            key = cv2.waitKey(120) & 0xFF
            if key == ord("s"):
                selected_frame = roi.copy()
                print(f"Frame {i} selected for measurement for particle {pid}.")
                break
            elif key == ord("q"):
                print(f"Skipping particle {pid}.")
                selected_frame = None
                break
            elif key == ord("e"):
                print("Exiting script by user request.")
                quit_all = True
                break
        if quit_all:
            break
        if selected_frame is None:
            continue

        measured_radius_px = measure_radius(selected_frame)
        if measured_radius_px is None:
            continue
        measured_radius_um = measured_radius_px * conversion_factor
        print(
            f"Measured radius for particle {pid} in folder '{folder}': {measured_radius_um:.2f} µm"
        )
        measurement = {
            "folder": folder,
            "particle_id": pid,
            "measured_radius_um": measured_radius_um,
        }
        results.append(measurement)
        # Append measurement to CSV immediately.
        df_new = pd.DataFrame([measurement])
        if os.path.exists(output_csv):
            df_new.to_csv(output_csv, mode="a", header=False, index=False)
        else:
            df_new.to_csv(output_csv, index=False)
        processed_particles.add((folder, pid))
        print("Press any key to continue to the next particle.")
        cv2.waitKey(0)
        cv2.destroyWindow("Zoomed Particle")
    cap.release()
    cv2.destroyAllWindows()
    if quit_all:
        break

print("\nMeasurements complete. Results saved to", output_csv)
