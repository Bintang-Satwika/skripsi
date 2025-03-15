from env_1 import FJSPEnv
import numpy as np
import random
from tqdm import tqdm
from MASKING_ACTION_MODEL import masking_action
import json
import os
# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)


def FAA(states, env):
    # Initialize a list for 3 agents (indices 0, 1, 2)
    fastest_agents_available = [False, False, False]
    
    # Iterate over agent indices in descending order (agent 3 is fastest at index 2)
    for i in sorted(range(len(states)), reverse=True):
        state = states[i]
        # Check if the agent is idle (status == 0)
        if state[env.state_status_location_all[i]] == 0:
            job_op = (state[env.state_first_job_operation_location[0]],
                      state[env.state_second_job_operation_location[0]])
            # Check if the required job operation is within the agent's capability
            if job_op in state[env.state_operation_capability_location]:
                fastest_agents_available[i] = True
                # Since we want only the fastest available agent,
                # we break as soon as one is found.
                break
    return fastest_agents_available

def hitl_action(states, env):
    # Get the baseline mask for each agent's actions.
    mask_actions = masking_action(states, env)
    print("mask_actions:\n", mask_actions)
    # Get FAA flags: a list where only the fastest available agent is True.
    fastest_agents_available = FAA(states, env)
    print("fastest_agents_available:\n", fastest_agents_available)
    
    # Look for the fastest available agent.
    for idx, available in enumerate(fastest_agents_available):
        if not available:
            # Override that agent's mask to force the Accept action.
            # Action space: [Accept, Wait yr-1, Wait yr-2, Decline, Continue]
            mask_actions[idx][0]= False
            # Since FAA returns only one fastest agent, we can break out of the loop.
            break
    print("mask_actions after:\n", mask_actions)

    return mask_actions





if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=30)
    rewards = {}
    makespan = {}
    for episode in range(1, 2):
        print("\nepisode:", episode)
        state, info = env.reset(seed=1000+episode)
        reward_satu_episode = 0
        done = False
        truncated = False
        #print("\nEpisode:", episode)
        #print("Initial state:", state)
        while not done and not truncated:
            print("\n")
            env.render()
            if len(env.conveyor.product_completed)>= env.n_jobs:
                print("All jobs are completed.")
                break

            hitl_actions=hitl_action(state, env)
            print("hitl_actions:", hitl_actions)
            actions=[]

            for action_agent in hitl_actions:
                true_indices = np.where(action_agent)[0]
                random_actions = random.choice(true_indices)
                actions.append(random_actions)

            actions = np.array(actions)
            print("actions:", actions)


            if None in actions:
                print("FAILED ACTION: ", actions)
                break


            next_state, reward, done, truncated, info = env.step(actions)
            reward = np.mean(reward)
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

        rewards[episode] = reward_satu_episode
        makespan[episode] = env.step_count
        
        print("env.conveyor.product_completed:", env.conveyor.product_completed)
        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count)
        order = {'A': 0, 'B': 1, 'C': 2}


    env.close()
    print("rewards:", rewards)
    # file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_FCFS_2_seed_op.json")
    # with open(file_path, "w") as f:
    #     json.dump(rewards, f, indent=4)
    # file_path= os.path.join(CURRENT_DIR, "Testing_makespan_FCFS_2_seed_op.json")
    # with open(file_path, "w") as f:
    #     json.dump(makespan, f, indent=4)

        #print("product sorted: ",sorted_jobs)
    print("selesai")