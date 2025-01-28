import gymnasium
import highway_env
import numpy as np
import pprint
# Membuat environment dengan konfigurasi multi-agent dan aksi kontinu
env = gymnasium.make(
    "highway-v0",
    render_mode="rgb_array",
    config={
        "vehicles_count": 1,
        "controlled_vehicles": 4,  # Dua kendaraan yang dikontrol
        "action": {
            "type": "ContinuousAction",
        }
    }
)
obs, info = env.reset()
pprint.pprint(env.unwrapped.config)
print(env.observation_space)
print(np.shape(obs))
print("action space")
print(env.action_space.shape)
# Loop utama
try:
    while True:
        # Aksi kontinu acak untuk setiap kendaraan yang dikontrol
        # Misalnya: longitudinal (percepatan) dan lateral (kemudi) dalam range [-1, 1]
        #actions = {agent_id: np.random.uniform(-1, 1, size=2) for agent_id in range(env.unwrapped.config["controlled_vehicles"])}
        action=[0,0]
        # Melakukan satu langkah di environment
        observations, rewards, dones, truncated, info = env.step(action)
        
        # Render environment
        env.render()
        
finally:
    env.close()
