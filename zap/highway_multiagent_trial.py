import gymnasium as gym
import highway_env
import pprint
import numpy as np
import random
env = gym.make(
  "highway-v0",
  render_mode="rgb_array",

)
env.unwrapped.config.update({
  "controlled_vehicles": 1,
   "vehicles_count": 3,
  "observation": {
    "type": "MultiAgentObservation",
    "observation_config": {
      "type": "Kinematics",
      "vehicles_count": 3,
      "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    }
  },
  "action": {
    "type": "MultiAgentAction",
    "action_config": {
      "type": "ContinuousAction",
    }
  }
})

env.reset(seed=0)

obs, info = env.reset()
pprint.pprint(env.unwrapped.config)
print(env.observation_space)
print(np.shape(obs))
#a=pd.DataFrame(np.array(obs).reshape(-1,7), columns=["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"])
#print(a)



while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    print("\n")
    #action =(random.randint(0,4),random.randint(0,4))
    action = tuple( [0,0] for obs_i in obs)
    print("action.shape",np.shape(action))
    obs, reward, done, truncated, info =  env.step(action)
    #a=pd.DataFrame(np.array(obs).reshape(-1,7), columns=["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"])
    #print(a)
    print(np.shape(obs))
    #print(action)
    #print(env.observation_space)
    env.render()