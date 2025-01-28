import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import random
import pprint
import pandas as pd
import numpy as np


config_env={
    "controlled_vehicles":3,
    "vehicles_count": 5,
    "screen_width": 1000,       # Set screen width to 800
    "screen_height": 200,       # Set screen height to 100
    "observation": {
      "type": "MultiAgentObservation",
      "observation_config": 
      {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": 
        {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
       "normalize":  True,
      }
    },

    
  "action": 
  {
    "type": "MultiAgentAction",
    "action_config": 
    {
      "type": "ContinuousAction",
    }
  }
}

env = gym.make('highway-v0', render_mode='rgb_array',   
                     config=config_env)

obs, info = env.reset()
pprint.pprint(env.unwrapped.config)
print(env.observation_space)
print(np.shape(obs))
print("action space")
print(env.action_space.shape)
#a=pd.DataFrame(np.array(obs).reshape(-1,7), columns=["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"])
#print(a)



while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    print("\n")
    #action =(random.randint(0,4),random.randint(0,4))
    action=([0,0],[0,0],[0,0])

    obs, reward, done, truncated, info = env.step(action)
    #a=pd.DataFrame(np.array(obs).reshape(-1,7), columns=["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"])
    #print(a)
    print(env.action_space)
    print(np.shape(obs))
    #print(action)
    #print(env.observation_space)
    env.render()