import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import pprint
from highway_env.vehicle.kinematics import Vehicle
env = gym.make("highway-fast-v0", config={"manual_control": True,} ,render_mode='rgb_array')
pprint.pprint(env.unwrapped.config)
while True:
  done = truncated = False
  obs, info = env.reset()
  iterasi = 0
  while not (done or truncated):
    iterasi += 1
    #action =(random.randint(0,4),random.randint(0,4))
    #action=(1,1,1)
    action=[0,0]
    obs, reward, done, truncated, info = env.step(action)
    obs = np.array(obs, dtype=np.float16)
    if iterasi %10 == 0:
        print("\n")
        print("reward",reward)
        print("done",done)
        print("truncated",truncated)
        print("info",info)
        #a=pd.DataFrame(np.array(obs).reshape(-1,len(columns_selected)), columns=columns_selected)

        #print(a.head(10))
        print(env.action_space)
        print(np.shape(obs))
  

    #print(action)
    #print(env.observation_space)
    env.render()