import numpy as np
import matplotlib.pyplot as plt

# List of CSV files to process
trapped_files = [
    "Data/TEK00026.CSV",
    "Data/TEK00027.CSV",
    "Data/TEK00028.CSV",
    "Data/TEK00029.CSV",
    "Data/TEK00030.CSV",
    "Data/TEK00031.CSV",
    "Data/TEK00032.CSV",
    "Data/TEK00033.CSV",
    "Data/TEK00034.CSV",
    "Data/TEK00035.CSV",
]


def select_slice_from_file(file_path):
    try:
        # Skip the first 15 rows, then read the header and data.
        # The header (column names) is assumed to be in the 16th row.
        data = np.genfromtxt(file_path, delimiter=",", skip_header=15, names=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # Extract the CH1 column
    try:
        ch1_data = data["CH1"]
    except Exception as e:
        print(f"Error accessing column 'CH1' in {file_path}: {e}")
        return

    # Create a plot for CH1 data
    fig, ax = plt.subplots()
    ax.plot(ch1_data, label="CH1")
    ax.set_title(f"Click two points (start & end) for:\n{file_path}")
    ax.set_xlabel("Index")
    ax.set_ylabel("CH1 Value")
    plt.legend()

    # Wait for two mouse clicks (click anywhere on the plot)
    pts = plt.ginput(2, timeout=-1)
    plt.close(fig)

    if len(pts) < 2:
        print("Not enough points were selected. Please try again.")
        return

    # Convert the x-coordinates of the clicks to integer indices
    start_idx, end_idx = sorted([int(round(pt[0])) for pt in pts])
    print(f"For file {file_path}, use slice: np.s_[{start_idx}:{end_idx}]")


# Loop through each file and allow the user to select slice indices
for file in trapped_files:
    select_slice_from_file(file)
