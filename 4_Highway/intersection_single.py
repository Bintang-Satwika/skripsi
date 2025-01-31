import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import pprint
from highway_env.vehicle.kinematics import Vehicle

Vehicle.MIN_SPEED = -5  # Atur kecepatan minimum menjadi -20 m/s
Vehicle.MAX_SPEED = 10   # Atur kecepatan maksimum menjadi 50 m/s
columns_selected = ["presence", "x", "y", "vx", "vy","heading", "ang_off", "sin_h", "cos_h", "sin_d","cos_d"]
config = {
      "vehicles_count": 0,
    "observation": {
        
        "type": "Kinematics",
        
        "vehicles_count": 0,
        "features": columns_selected,
        # "features_range": {
        #     "x": [-100, 100],
        #     "y": [-100, 100],
        #     "vx": [-100, 100],
        #     "vy": [-100, 100]
        # },
        "absolute": False,
        "order": "sorted",
        "normalize": False,
        "clip": True,
        "include_obstacle": True,
        "observe_intentions": True,
    },
    "action": {
        #"type": "DiscreteMetaAction"
        "type": "ContinuousAction"
    },
    'show_trajectories': True,
    "manual_control": True,
    "reward_speed_range": [20, 30],
    "collision_reward": -1,
    "duration": 50,
    "initial_vehicle_count": 0,
}

env = gym.make('intersection-v0', render_mode='rgb_array', config=config)
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