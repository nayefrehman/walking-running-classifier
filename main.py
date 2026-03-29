import os
import pandas as pd
import h5py
import matplotlib.pyplot as plt

data_folder = './rawData/'

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
fig, ax = plt.subplots() 
#Graph acceleration vs time
ax.plot(df["Time"], df["Ax"], label="Ax")
ax.plot(df["Time"], df["Ay"], label="Ay")
ax.plot(df["Time"], df["Az"], label="Az")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration")
ax.set_title("Sample Acceleration Data")
ax.legend()

plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# scatter instead of line
sc = ax.scatter(df["Ax"], df["Ay"], df["Az"], c=df["Time"])

ax.set_xlabel("Ax")
ax.set_ylabel("Ay")
ax.set_zlabel("Az")
ax.set_title("3D Acceleration (colored by time)")

plt.colorbar(sc, label="Time")

plt.show()