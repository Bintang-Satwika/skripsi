import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import pprint

# Konfigurasi environment dengan Lidar Observation
config = {
    "observation": {
        "type": "LidarObservation",
        "cells": 36,             # Jumlah sektor (default: 36 sektor, satu sektor = 10 derajat)
        "maximum_range": 30.0,   # Jarak maksimum yang bisa diamati [meter]
        "normalize": False        # Normalisasi jarak dalam [0, 1]
    },
    "action": {
        "type": "ContinuousAction"
    },
    "lanes_count": 4,
    "show_trajectories": True,
    "manual_control": True,
    "reward_speed_range": [20, 30],
    "collision_reward": -1,
}

# Buat environment
env = gym.make('highway-v0', render_mode='rgb_array', config=config)

# Reset environment dan print bentuk observasi awal
obs, info = env.reset()
print("Shape of Lidar Observation:", obs.shape)
print("Lidar Observation Example (per sektor):")
print(np.round(obs, 2))  # Tampilkan nilai observasi lidar

pprint.pprint(env.unwrapped.config)
print(env.action_space.shape)
print("-----------------")

# Loop utama
while True:
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        print("\n")
        
        # Aksi manual (ubah jika menggunakan model RL)
        action = [0, 0]  # Kendaraan tetap di jalur
        obs, reward, done, truncated, info = env.step(action)
        
        # Print observasi Lidar
        print("Lidar Observation (36 sektor):")
        print(np.round(obs, 2))  # Tampilkan jarak dan kecepatan relatif
        
        # Print informasi reward dan status
        print("reward:", reward)
        print("done:", done)
        print("truncated:", truncated)
        print("info:", info)
        
        # Render environment
        env.render()
