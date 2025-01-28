import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import pprint
from highway_env.vehicle.kinematics import Vehicle
Vehicle.MIN_SPEED = -5  # Atur kecepatan minimum menjadi -20 m/s
Vehicle.MAX_SPEED = 20  # Atur kecepatan maksimum menjadi 50 m/s
columns_selected = ["presence", "x", "y", "vx", "vy", "sin_h", "cos_h"]
config = {
       "vehicles_count": 10,
    "observation": {
        
        "type": "Kinematics",
        
        "vehicles_count": 10,
        "features": columns_selected,
        "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-5, 20],
                "vy": [-5, 20]
            },
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
    "lanes_count": 4,
    'show_trajectories': False,
    "manual_control": False,
    #"reward_speed_range": [5, 20],
    "collision_reward": -1,
    "duration": 100,
}

env = gym.make('highway-fast-v0', render_mode='rgb_array', config=config)
while True:
  done = truncated = False
  state, info = env.reset()
  iterasi = 0
  while not (done or truncated):
    iterasi += 1
    #action =(random.randint(0,4),random.randint(0,4))
    #action=(1,1,1)
    action=[0,0]
    next_state, reward, done, truncated, info = env.step(action)
    print("next_state",next_state.shape)
    #obs = np.array(obs, dtype=np.float16)
    # if iterasi %10 == 0:
    #     print("\n")
    #     print("reward",reward)
    #     print("done",done)
    #     print("truncated",truncated)
    #     print("info",info)
    #     a=pd.DataFrame(np.array(obs).reshape(-1,len(columns_selected)), columns=columns_selected)

    #     print(a.head(10))
    #     print(env.action_space)
    #     print(np.shape(obs))
    #print(action)
    #print(env.observation_space)
    env.render()