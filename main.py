import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
#step 4 library
from scipy.signal import butter, filtfilt


data_folder = "rawData"

if not os.path.exists(data_folder):
    print("Folder not found:", data_folder)
    exit()

print("Starting storage process...")
print("Members found:", os.listdir(data_folder))

with h5py.File("project_data.h5", "w") as hdf5_file:

    raw_group = hdf5_file.create_group("raw")

    for member in os.listdir(data_folder):

        member_path = os.path.join(data_folder, member)

        if os.path.isdir(member_path):

            print("\nProcessing member:", member)

            member_group = raw_group.create_group(member)

            for file in os.listdir(member_path):

                if file.endswith(".csv"):

                    file_path = os.path.join(member_path, file)
                    print("Reading file:", file_path)

                    df = pd.read_csv(file_path)

                    df = df[
                        [
                            "Time (s)",
                            "Acceleration x (m/s^2)",
                            "Acceleration y (m/s^2)",
                            "Acceleration z (m/s^2)",
                        ]
                    ]
                    #renaming to shorter and clearer names
                    df.columns = ["Time", "Ax", "Ay", "Az"]

                    dataset_name = file.replace(".csv", "")
                    member_group.create_dataset(dataset_name, data=df.values)

print("\nHDF5 file created.")


#Step 3

def compute_magnitude(df):
    return np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)

def plot_acceleration_vs_time(ax, df, title, metadata):
    ax.plot(df["Time"], df["Ax"], label="Ax")
    ax.plot(df["Time"], df["Ay"], label="Ay")
    ax.plot(df["Time"], df["Az"], label="Az")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_title(f"{title}\n{metadata}")
    ax.legend(fontsize=6)

def plot_bubble_chart(ax, df, title, metadata):
    df_5s = df[df["Time"] <= 5]
    ax.scatter(df_5s["Time"], df_5s["Magnitude"], s=df_5s["Magnitude"], alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"{title}\n{metadata}")

with h5py.File("project_data.h5", "r") as hdf5_file:
    raw = hdf5_file["raw"]

    for member in raw.keys():
        print("Plotting for:", member)
        files = list(raw[member].keys())

        fig, ax = plt.subplots(6, 2, figsize=(12, 18))
        fig.suptitle(f"Member: {member}", fontsize=16)

        for i, file in enumerate(files[:6]):
            dataset = raw[member][file]
            data = dataset[:]

            df = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])
            df["Magnitude"] = compute_magnitude(df)

            #Meta data: file size
            shape = dataset.shape
            size_kb = dataset.nbytes / 1024
            metadata = f"Shape: {shape} | Size: {size_kb:.1f} KB"

            #acceleration vs time graph on left
            plot_acceleration_vs_time(
                ax[i, 0],
                df,
                title=f"{file} - Acceleration",
                metadata=metadata
            )

            #Bubble chart on the right
            plot_bubble_chart(
                ax[i, 1],
                df,
                title=f"{file} - Bubble",
                metadata=metadata
            )

        #plt.tight_layout()
        #plt.show()
        
#Step 4
WINDOW_SIZE = 5
HIGHPASS_CUTOFF = 0.25
SAMPLE_RATE = 100

def fill_missing(df):
    return df.ffill().bfill()
#window size function
def apply_sma(df, window_size=WINDOW_SIZE):
    smoothed = df.copy()
    for col in ["Ax", "Ay", "Az"]:
        smoothed[col] = df[col].rolling(window=window_size, center=True).mean()
    return smoothed.ffill().bfill()

#high-pass filter
def apply_highpass(df, cutoff=HIGHPASS_CUTOFF, fs=SAMPLE_RATE, order=4):
    filtered = df.copy()

    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="high", analog=False)

    for col in ["Ax", "Ay", "Az"]:
        mean_val = df[col].mean()              #save original mean so it doesnt merge all axis together
        filtered[col] = filtfilt(b, a, df[col].values)
        filtered[col] += mean_val              #bring back the original mean

    return filtered

# Dictionary to hold plot samples per member: member -> list of (filename, raw_df, final_df)
# final_df = SMA then high-pass applied in sequence — this is what gets saved and plotted
member_plot_data = {}

# Open HDF5 in append mode to add the preprocessed group alongside raw
with h5py.File("project_data.h5", "a") as hdf5_file:

    # Create the top-level preprocessed group
    preprocessed_group = hdf5_file.create_group("preprocessed")

    # Loop through each member's folder
    for member in os.listdir(data_folder):
        member_path = os.path.join(data_folder, member)

        if os.path.isdir(member_path):
            print(f"Preprocessing: {member}")

            # Create a subgroup for this member under preprocessed
            member_group = preprocessed_group.create_group(member)
            member_plot_data[member] = []

            # Loop through each CSV file and apply preprocessing
            for file in os.listdir(member_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(member_path, file)

                    df_raw = pd.read_csv(file_path)
                    df_raw = df_raw[[
                        "Time (s)",
                        "Acceleration x (m/s^2)",
                        "Acceleration y (m/s^2)",
                        "Acceleration z (m/s^2)",
                    ]]
                    df_raw.columns = ["Time", "Ax", "Ay", "Az"]

                    # Step 4a: Fill any missing values
                    df_filled = fill_missing(df_raw)

                    # Step 4b: Apply SMA to reduce high-frequency noise
                    df_smoothed = apply_sma(df_filled)

                    # Step 4c: Apply high-pass filter directly on the SMA result
                    # The two filters are chained — high-pass receives SMA output as its input
                    df_final = apply_highpass(df_smoothed)

                    # Save the fully processed data (SMA + high-pass combined) into HDF5
                    dataset_name = file.replace(".csv", "")
                    member_group.create_dataset(dataset_name, data=df_final.values)

                    # Collect up to 6 files per member for plotting
                    if len(member_plot_data[member]) < 6:
                        member_plot_data[member].append((dataset_name, df_filled, df_final))

print("Preprocessed data saved to HDF5.")

#Raw and proccessed data algorithms
colors = {"Ax": "red", "Ay": "green", "Az": "blue"}

for member, samples in member_plot_data.items():

    num_files = len(samples)  # up to 6 files
    fig, axes = plt.subplots(num_files, 2, figsize=(14, 3.5 * num_files))
    fig.suptitle(
        f"Step 4 - Pre-Processing | Member: {member}\n"
        f"Left = Raw  |  Right = SMA + High-Pass Combined  |  First 5s Shown",
        fontsize=13,
        fontweight="bold"
    )

    for row, (name, raw_df, final_df) in enumerate(samples):

        #Limit to first 5 seconds
        raw_5s   = raw_df[raw_df["Time"] <= 5]
        final_5s = final_df[final_df["Time"] <= 5]

        #Edge cases
        ax_raw   = axes[row, 0] if num_files > 1 else axes[0]
        ax_final = axes[row, 1] if num_files > 1 else axes[1]

        #Raw data on the left
        for col, color in colors.items():
            ax_raw.plot(raw_5s["Time"], raw_5s[col], color=color, alpha=0.6, label=col)
        ax_raw.set_title(f"{name} — Raw (First 5s)")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Acceleration")
        ax_raw.legend(fontsize=7)
        ax_raw.grid(True)

        #Processed data on the right
        for col, color in colors.items():
            ax_final.plot(final_5s["Time"], final_5s[col], color=color, linewidth=1.5, label=col)
        ax_final.set_title(f"{name} — SMA + High-Pass Combined (First 5s)")
        ax_final.set_xlabel("Time (s)")
        ax_final.set_ylabel("Acceleration")
        ax_final.legend(fontsize=7)
        ax_final.grid(True)

    #plt.tight_layout()
    #plt.show()


# Step 5 - Feature Extraction & Normalization
# -----------------------------------------------------------------------

from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

WINDOW_SECONDS = 5    # Each segment is 5 seconds long
SAMPLES_PER_SEC = 100 # Must match SAMPLE_RATE from Step 4
SAMPLES_PER_WINDOW = WINDOW_SECONDS * SAMPLES_PER_SEC  # = 500 samples per window

def extract_features(window_df):
    """Extract 12 features per axis (Ax, Ay, Az) from a 5-second window.
    Each axis gets its own set of features, giving 36 features total per window.
    Features: mean, std, min, max, range, median, variance, RMS, skewness, kurtosis, energy, zero_crossings"""
    features = []
    for col in ["Ax", "Ay", "Az"]:
        signal = window_df[col].values

        mean        = np.mean(signal)
        std         = np.std(signal)
        minimum     = np.min(signal)
        maximum     = np.max(signal)
        sig_range   = maximum - minimum
        median      = np.median(signal)
        variance    = np.var(signal)
        rms         = np.sqrt(np.mean(signal**2))           # Root Mean Square — good for activity intensity
        skewness    = skew(signal)                           # Asymmetry of the signal distribution
        kurt        = kurtosis(signal)                      # Peakedness of the signal distribution
        energy      = np.sum(signal**2) / len(signal)       # Average signal energy per sample
        # Zero crossings: counts how often the signal crosses zero — higher for jumping
        zero_cross  = np.sum(np.diff(np.sign(signal - mean)) != 0)

        features.extend([mean, std, minimum, maximum, sig_range,
                          median, variance, rms, skewness, kurt, energy, zero_cross])
    return features

# Feature names for reference (12 features x 3 axes = 36 total)
feature_names = [
    f"{feat}_{axis}"
    for axis in ["Ax", "Ay", "Az"]
    for feat in ["mean", "std", "min", "max", "range", "median",
                 "variance", "rms", "skewness", "kurtosis", "energy", "zero_crossings"]
]

all_features = []   # Will hold all feature vectors across all members and files
all_labels   = []   # Corresponding label: "Walk" or "Jump"

# Open HDF5 to read preprocessed data and write segmented data
with h5py.File("project_data.h5", "a") as hdf5_file:

    # Create the segmented group to store 5-second windows
    if "segmented" in hdf5_file:
        del hdf5_file["segmented"]
    segmented_group = hdf5_file.create_group("segmented")

    preprocessed = hdf5_file["preprocessed"]

    for member in preprocessed.keys():
        print(f"\nExtracting features for: {member}")
        member_seg_group = segmented_group.create_group(member)

        for dataset_name in preprocessed[member].keys():
            data = preprocessed[member][dataset_name][:]
            df   = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])

            # Determine the label from the filename
            # Expects filenames to contain "walk" or "jump" (case-insensitive)
            name_lower = dataset_name.lower()
            if "walk" in name_lower:
                label = "Walk"
            elif "jump" in name_lower:
                label = "Jump"
            else:
                print(f"  Skipping {dataset_name} — could not determine label from filename")
                continue

            # Segment the signal into non-overlapping 5-second windows
            num_windows = len(df) // SAMPLES_PER_WINDOW
            if num_windows == 0:
                print(f"  Skipping {dataset_name} — not enough data for one 5s window")
                continue

            print(f"  {dataset_name} ({label}): {num_windows} windows")
            file_seg_group = member_seg_group.create_group(dataset_name)

            for i in range(num_windows):
                start = i * SAMPLES_PER_WINDOW
                end   = start + SAMPLES_PER_WINDOW
                window_df = df.iloc[start:end]

                # Save the raw window into the HDF5 segmented group
                file_seg_group.create_dataset(f"window_{i}", data=window_df.values)

                # Extract features from this window and store with its label
                features = extract_features(window_df)
                all_features.append(features)
                all_labels.append(label)

print(f"\nTotal windows extracted: {len(all_features)}")
print(f"Label breakdown — Walk: {all_labels.count('Walk')} | Jump: {all_labels.count('Jump')}")

# Convert to numpy arrays for normalization and model training
X = np.array(all_features)   # Shape: (num_windows, 36)
y = np.array(all_labels)      # Shape: (num_windows,)

# --- Normalization: Z-score standardization ---
# Ensures no single feature dominates due to scale differences
# Fit the scaler on ALL data here — train/test split happens in Step 6
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print(f"\nFeature matrix shape: {X_normalized.shape}")
print(f"Features per window:  {X_normalized.shape[1]} ({len(['Ax','Ay','Az'])} axes x {X_normalized.shape[1]//3} features each)")

# --- Visualization: Feature distributions per class ---
# Plot the mean of each feature grouped by Walk vs Jump so we can see which features separate the classes
walk_features = X_normalized[y == "Walk"]
jump_features = X_normalized[y == "Jump"]

walk_means = np.mean(walk_features, axis=0)
jump_means = np.mean(jump_features, axis=0)

fig, ax = plt.subplots(figsize=(18, 5))
x_pos = np.arange(len(feature_names))
width = 0.4

ax.bar(x_pos - width/2, walk_means, width, label="Walk", color="steelblue",  alpha=0.8)
ax.bar(x_pos + width/2, jump_means, width, label="Jump", color="darkorange", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
ax.set_ylabel("Normalized Mean Value")
ax.set_title("Step 5 - Mean Normalized Feature Values: Walk vs Jump")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#step 6
