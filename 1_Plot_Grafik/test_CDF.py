import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Directories
yr0_dir = 'D:\\KULIAH\\skripsi\\CODE\\skripsi\\3_FJSP_old\\tanpawait'
yr1_dir = 'D:\\KULIAH\\skripsi\\CODE\\skripsi\\3_FJSP_old\\yr1'
yr2_dir = 'D:\\KULIAH\\skripsi\\CODE\\skripsi\\3_FJSP_old\\yr2'
yr3_dir = 'D:\\KULIAH\\skripsi\\CODE\\skripsi\\3_FJSP_old\\yr3'

# File paths and labels
files = [
    os.path.join(yr0_dir, 'Testing_DQN_tanpawait_200ep_2.json'),
    os.path.join(yr1_dir, 'Testing_DQN_yr1_200ep.json'),
    os.path.join(yr2_dir, 'Testing_DQN_yr2_200ep.json'),
    os.path.join(yr3_dir, 'Testing_DQN_yr3_200ep.json'),
    os.path.join(yr2_dir, 'Testing_FCFS_200ep.json'),
    os.path.join(yr2_dir, 'Testing_FAA_200ep.json'),
]
labels = [
    'DQN\n(window = 1)',
    'DQN\n(window = 2)',
    'DQN\n(window = 3)',
    'DQN\n(window = 4)',
    'FCFS',
    'FAA'
]

# Plot setup
plt.figure()

# Loop through each file and plot its energy CDF
for filepath, label in zip(files, labels):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        energies = np.array(list(data['makespan'].values()), dtype=float)
        sorted_energies = np.sort(energies)
        cdf = np.arange(1, len(sorted_energies) + 1) / len(sorted_energies)
        plt.plot(sorted_energies, cdf, label=label)
    else:
        print(f"[WARNING] File not found: {filepath}")

# Final touches
plt.xlabel('value')
plt.ylabel('probabilitas')
plt.title('CDF of 200 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
