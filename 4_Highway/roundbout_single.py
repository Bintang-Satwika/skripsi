import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import pprint
from highway_env.vehicle.kinematics import Vehicle

Vehicle.MIN_SPEED = -2  # Atur kecepatan minimum menjadi -20 m/s
Vehicle.MAX_SPEED = 10   # Atur kecepatan maksimum menjadi 50 m/s
columns_selected = ["presence", "x", "y", "vx", "vy"]
config = {
    "observation": {
        "type": "Kinematics",
        "normalize": True,
        # "features_range": {
        #     #"x": [-100, 100],
        #     #"y": [-100, 100],
        #     #"vx": [-100, 100],
        #     #"vy": [-100, 100]
        # },
    },
    "action": {
        #"type": "DiscreteMetaAction"
        "type": "ContinuousAction"
    },
    "incoming_vehicle_destination": None,
    "duration": 30, # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 600,  # [px] height of the pygame window
    "centering_position": [0.5, 0.6],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    'show_trajectories': True,
    "manual_control": True,
     "reward_speed_range": [20, 30],
}

env = gym.make("roundabout-v0", render_mode='rgb_array', config=config)
obs, info = env.reset()
print("info",info)
print(obs.shape)
a=pd.DataFrame(np.array(obs).reshape(-1,len(columns_selected)), columns=columns_selected)
print(a)
pprint.pprint(env.unwrapped.config)
print(env.action_space.shape)
print("-----------------")
import time
while True:
  done = truncated = False
  obs, info = env.reset()
  iterasi = 0
  while not (done or truncated):
    iterasi += 1
    #action =(random.randint(0,4),random.randint(0,4))
    #action=(1,1,1)
    action=[1,2]
    #action=0
    obs, reward, done, truncated, info = env.step(action)
    obs = np.array(obs, dtype=np.float16)
    if iterasi %10 == 0:
        print("\n")
        print("reward",reward)
        print("done",done)
        print("truncated",truncated)
        print("info",info)
        a=pd.DataFrame(np.array(obs).reshape(-1,len(columns_selected)), columns=columns_selected)

        print(a.head(10))
        print(env.action_space)
        print(np.shape(obs))
    #time.sleep(0.01)

    #print(action)
    #print(env.observation_space)
    env.render()