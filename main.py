import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
#step 4 library
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json


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


# Step 5: Feature Extraction & Normalization 
#Feature extraction function
def extract_features(df):
    """
    Extract ≥10 features from Ax, Ay, Az, and Magnitude.
    Returns a dictionary of features.
    """
    features = {}
    axes = ["Ax", "Ay", "Az"]
    df["Magnitude"] = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
    axes.append("Magnitude")
    
    for axis in axes:
        col = df[axis]
        features[f"{axis}_mean"] = col.mean()
        features[f"{axis}_median"] = col.median()
        features[f"{axis}_std"] = col.std()
        features[f"{axis}_var"] = col.var()
        features[f"{axis}_max"] = col.max()
        features[f"{axis}_min"] = col.min()
        features[f"{axis}_range"] = col.max() - col.min()
        features[f"{axis}_rms"] = np.sqrt(np.mean(col**2))
        features[f"{axis}_skew"] = skew(col)
        features[f"{axis}_kurtosis"] = kurtosis(col)
    return features

#Normalization function
def normalize_features(feature_dict, method="minmax"):
    """
    Normalize features using Min-Max scaling (default) or z-score.
    """
    keys = list(feature_dict.keys())
    values = np.array([feature_dict[k] for k in keys]).reshape(-1, 1)

    if method == "minmax":
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(values).flatten()
    elif method == "zscore":
        normalized = (values - values.mean()) / values.std()
        normalized = normalized.flatten()
    else:
        raise ValueError("Normalization method must be 'minmax' or 'zscore'")

    normalized_dict = {k: v for k, v in zip(keys, normalized)}
    return normalized_dict

#plot
def plot_features(features, title, ax=None):
    keys = list(features.keys())
    values = [features[k] for k in keys]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(range(len(values)), values, color="skyblue")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Value")
    ax.set_title(title)
    ax.grid(True)

for member, samples in member_plot_data.items():
    print(f"\nFeature extraction & normalization for member: {member}")
    
    num_files = len(samples)
    fig, axes = plt.subplots(num_files, 1, figsize=(14, 4*num_files))
    
    if num_files == 1:
        axes = [axes]

    for idx, (name, raw_df, final_df) in enumerate(samples):
        # Extract features from 5-second preprocessed data
        features = extract_features(final_df)
        normalized_features = normalize_features(features, method="minmax")
        
        # Plot normalized features
        plot_features(normalized_features, title=f"{member} | {name} | Normalized Features", ax=axes[idx])

    plt.tight_layout()
    plt.show()

# Step 6 - Logistic Regression Classifier with Training Curves

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import pickle
import json

WINDOW_SECONDS = 5
SAMPLE_RATE = 100
SAMPLES_PER_WINDOW = WINDOW_SECONDS * SAMPLE_RATE
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def segment_data(df, label):
    segments = []
    num_windows = len(df) // SAMPLES_PER_WINDOW

    for i in range(num_windows):
        start = i * SAMPLES_PER_WINDOW
        end = start + SAMPLES_PER_WINDOW
        segment = df.iloc[start:end].copy()
        segments.append((segment, label))

    return segments

def get_label(filename):
    filename = filename.lower()
    if "walk" in filename:
        return 0
    elif "jump" in filename:
        return 1
    return None

# ---------------------------------------------------
# 1. Split FILES by class first (prevents leakage)
# ---------------------------------------------------
walk_files = []
jump_files = []

with h5py.File("project_data.h5", "r") as f:
    preprocessed = f["preprocessed"]

    for member in preprocessed.keys():
        for name in preprocessed[member].keys():
            label = get_label(name)
            if label == 0:
                walk_files.append((member, name))
            elif label == 1:
                jump_files.append((member, name))

np.random.shuffle(walk_files)
np.random.shuffle(jump_files)

# 80% train, 10% validation, 10% test
n_walk = len(walk_files)
n_jump = len(jump_files)

train_walk_end = int(0.8 * n_walk)
val_walk_end = int(0.9 * n_walk)

train_jump_end = int(0.8 * n_jump)
val_jump_end = int(0.9 * n_jump)

train_files = walk_files[:train_walk_end] + jump_files[:train_jump_end]
val_files   = walk_files[train_walk_end:val_walk_end] + jump_files[train_jump_end:val_jump_end]
test_files  = walk_files[val_walk_end:] + jump_files[val_jump_end:]

np.random.shuffle(train_files)
np.random.shuffle(val_files)
np.random.shuffle(test_files)

print("Train files:", len(train_files))
print("Validation files:", len(val_files))
print("Test files:", len(test_files))

# ---------------------------------------------------
# 2. Convert files into 5-second segments
# ---------------------------------------------------
train_segments = []
val_segments = []
test_segments = []

with h5py.File("project_data.h5", "r") as f:
    preprocessed = f["preprocessed"]

    for member, name in train_files:
        data = preprocessed[member][name][:]
        df = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])
        label = get_label(name)
        train_segments.extend(segment_data(df, label))

    for member, name in val_files:
        data = preprocessed[member][name][:]
        df = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])
        label = get_label(name)
        val_segments.extend(segment_data(df, label))

    for member, name in test_files:
        data = preprocessed[member][name][:]
        df = pd.DataFrame(data, columns=["Time", "Ax", "Ay", "Az"])
        label = get_label(name)
        test_segments.extend(segment_data(df, label))

print("Train segments:", len(train_segments)) 
print("Validation segments:", len(val_segments)) 
print("Test segments:", len(test_segments)) 

np.random.shuffle(train_segments)
np.random.shuffle(val_segments)
np.random.shuffle(test_segments)


# ---------------------------------------------------
# SAVE SEGMENTED DATA TO HDF5 (REQUIRED)
# ---------------------------------------------------
with h5py.File("project_data.h5", "a") as f:
    segmented_group = f.require_group("segmented")

    train_group = segmented_group.require_group("train")
    val_group = segmented_group.require_group("val")
    test_group = segmented_group.require_group("test")

    # Train
    for i, (seg_df, label) in enumerate(train_segments):
        dset = train_group.create_dataset(f"seg_{i}", data=seg_df.values)
        dset.attrs["label"] = label

    # Validation
    for i, (seg_df, label) in enumerate(val_segments):
        dset = val_group.create_dataset(f"seg_{i}", data=seg_df.values)
        dset.attrs["label"] = label

    # Test
    for i, (seg_df, label) in enumerate(test_segments):
        dset = test_group.create_dataset(f"seg_{i}", data=seg_df.values)
        dset.attrs["label"] = label

print("Segmented data saved to HDF5.")

# ---------------------------------------------------
# 3. Feature extraction
# ---------------------------------------------------
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

# Get consistent feature order from one sample
sample_df, _ = train_segments[0]
feature_names = list(extract_features(sample_df.copy()).keys())

# Train set
for seg_df, label in train_segments:
    features = extract_features(seg_df.copy())
    X_train.append([features[k] for k in feature_names])
    y_train.append(label)

# Validation set
for seg_df, label in val_segments:
    features = extract_features(seg_df.copy())
    X_val.append([features[k] for k in feature_names])
    y_val.append(label)

# Test set
for seg_df, label in test_segments:
    features = extract_features(seg_df.copy())
    X_test.append([features[k] for k in feature_names])
    y_test.append(label)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print("\nClass distribution:")
print("Train -> Walking:", np.sum(y_train == 0), "Jumping:", np.sum(y_train == 1))
print("Val   -> Walking:", np.sum(y_val == 0), "Jumping:", np.sum(y_val == 1))
print("Test  -> Walking:", np.sum(y_test == 0), "Jumping:", np.sum(y_test == 1))

# ---------------------------------------------------
# 4. Normalize using ONLY training data
# ---------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# 5. Train logistic regression epoch-by-epoch
# ---------------------------------------------------
epochs = 12
model = SGDClassifier(
    loss="log_loss",      # logistic regression
    max_iter=1,
    tol=None,
    random_state=RANDOM_SEED
)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

classes = np.array([0, 1])

for epoch in range(epochs):
    # shuffle training data each epoch
    indices = np.random.permutation(len(X_train))
    X_train_epoch = X_train[indices]
    y_train_epoch = y_train[indices]

    if epoch == 0:
        model.partial_fit(X_train_epoch, y_train_epoch, classes=classes)
    else:
        model.partial_fit(X_train_epoch, y_train_epoch)

    # predicted probabilities for loss
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)

    # predicted labels for accuracy
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

# ---------------------------------------------------
# 6. Final evaluation on test set
# ---------------------------------------------------
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print("\nFinal Test Accuracy:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Walking", "Jumping"]))

print("\nConfusion Matrix:")
print(cm)

# ---------------------------------------------------
# 7. Plot training curves
# ---------------------------------------------------
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_loss_history, label="Training Loss")
plt.plot(epochs_range, val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training Curve - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_acc_history, label="Training Accuracy")
plt.plot(epochs_range, val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Training Curve - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 8. Save results
# ---------------------------------------------------
with open("step6_results.json", "w") as f:
    json.dump({
        "test_accuracy": float(test_accuracy),
        "train_loss_history": [float(x) for x in train_loss_history],
        "val_loss_history": [float(x) for x in val_loss_history],
        "train_acc_history": [float(x) for x in train_acc_history],
        "val_acc_history": [float(x) for x in val_acc_history],
        "confusion_matrix": cm.tolist()
    }, f, indent=4)

with open("step6_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler
    }, f)

print("\nStep 6 completed properly: model trained, training curves recorded, and test accuracy saved.")
