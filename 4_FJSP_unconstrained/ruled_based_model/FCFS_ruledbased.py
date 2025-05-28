from env_80 import FJSPEnv
import numpy as np
import random
from tqdm import tqdm

import json
import os
import numpy as np
import random

from RULED_BASED import MASKING_action,  FCFS_action, RANDOM_action, HITL_action
# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if __name__ == "__main__":
    env = FJSPEnv(window_size=2, num_agents=3, max_steps=1000, episode=1)
    rewards = {}
    makespan = {}
    energy= {}
    success = {}
    success_energy = {}
    episode_seed=0
    for episode in range(1, 1000+1):
        print("\nepisode:", episode)
        state, info = env.reset(seed=1000+episode)
        if (episode-1) %1 == 0:
            episode_seed= episode-1

        env.conveyor.episode_seed= episode_seed
        print(env.conveyor.episode_seed)
        reward_satu_episode = 0
        done = False
        truncated = False
        #print("\nEpisode:", episode)
        #print("Initial state:", state)
        while not done and not truncated: 
            if len(env.conveyor.product_completed)>= env.n_jobs:
                print("All jobs are completed.")
                break

            #actions, masking=HITL_action(state, env)
            actions=FCFS_action(state, env)



            if None in actions:
                print("FAILED ACTION: ", actions)
                break


            next_state, reward, done, truncated, info = env.step(actions)
            # print("state:\n", state)
            # print("actions:", actions)
            # print("next_state:\n", next_state)
            # env.render()
            # print("\n")
            reward = np.mean(reward)
            #reward= reward[0]
            reward_satu_episode += reward
            
            if env.FAILED_ACTION:
                print("episode:", episode)
                print("state:\n", state)
                print("actions:", actions)
                print("next_state:\n", next_state)
                #print(env.observation_all)
                #print("info:", info)
                print("FAILED ENV")
                break

            state = next_state
            #print("next_state:", next_state)

        # if  None in actions:
        #     break

        if len(env.conveyor.product_completed) <21 or  env.step_count >= 250:
            success[episode] = 0
        else:
            success[episode] = 1
        if len(env.conveyor.product_completed) <21 or  env.step_count >= 250 or sum(env.agents_energy_consumption)>=250:
            success_energy[episode] = 0
        else:
            success_energy[episode] = 1

        rewards[episode] = reward_satu_episode
        makespan[episode] = env.step_count
        energy[episode] = sum(env.agents_energy_consumption)
        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count, "total product completed:", len(env.conveyor.product_completed))
        print("product completed:", len(env.conveyor.product_completed), "success:", success[episode])
        order = {'A': 0, 'B': 1, 'C': 2}
        print("")
        combined_data = {
        "rewards": rewards,
        "makespan": makespan,
        "energy": energy,
         "success": success,
         "success_energy": success_energy,}

        # Write the combined dictionary to a single JSON file
        file_path = os.path.join(CURRENT_DIR, "Testing_FCFS_1000ep_80.json")
        with open(file_path, "w") as f:
            json.dump(combined_data, f, indent=4)



    env.close()
    print("rewards:", np.mean(list(rewards.values())))
    print("makespan:",  np.mean(list(makespan.values())))
        #print("product sorted: ",sorted_jobs)
    print("selesai")