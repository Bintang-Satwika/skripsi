import gymnasium as gym
import highway_env
import numpy as np
import pprint

# Konfigurasi environment dengan Occupancy Grid Observation
config = {
    "vehicles_count": 15,
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "lanes_count": 2,
    "show_trajectories": True,
    "manual_control": True,
    "reward_speed_range": [20, 30],
    "collision_reward": -1,
}

# Buat environment
env = gym.make('highway-v0', render_mode='rgb_array', config=config)

# Reset environment dan print bentuk observasi awal
obs, info = env.reset()
print("Shape of Occupancy Grid Observation:", obs.shape)
print("Occupancy Grid Observation Example (grid values):")
print(np.round(obs, 2))  # Tampilkan nilai occupancy grid

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
        
        # Print observasi Occupancy Grid
        print("Occupancy Grid Observation (grid values):")
        print(np.round(obs, 2))  # Tampilkan grid occupancy
        
        # Print informasi reward dan status
        print("reward:", reward)
        print("done:", done)
        print("truncated:", truncated)
        print("info:", info)
        
        # Render environment
        env.render()
