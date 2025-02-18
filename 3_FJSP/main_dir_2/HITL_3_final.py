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
    '''
    1. keika ada job di yr-1, maka agent dapat memilih action wait yr-1.
    2. FAA akan meng-cancel action wait yr-1 pada agent tersebut bilamana agent lain lebih cepat.
    3. Kondisi:
        a. Jika agent 1 idle dan operasi yr-1 bisa dikerjakan agent 1, maka agent 2 dan 3 akan cancel action wait yr-1.
        b. Jika agent 2 idle dan operasi yr-1 bisa dikerjakan agent 2, maka agent 1 dan 3 akan cancel action wait yr-1.
        c. Jika agent 3 idle dan operasi yr-1 bisa dikerjakan agent 3, maka agent 1 dan 2 akan cancel action wait yr-1.
    '''
    fastest_agents_available = np.full((3, 3), False)
    job_op_1 = (states[0, env.state_first_job_operation_location[1]],
                  states[0, env.state_second_job_operation_location[1]])
    job_op_2 = (states[1, env.state_first_job_operation_location[1]], 
                states[1, env.state_second_job_operation_location[1]])
    job_op_3 = (states[2, env.state_first_job_operation_location[1]],
                states[2, env.state_second_job_operation_location[1]])
    
    # Check Agent 3 (fastest; index 2)
    if states[2, env.state_status_location_all[2]] == 0:
        # Agent 3 is available and capable
        if np.all(np.isin(job_op_3, states[2, env.state_operation_capability_location])):
            fastest_agents_available[2,2] = True  
        elif np.all(np.isin(job_op_2, states[2, env.state_operation_capability_location])):
            fastest_agents_available[2,1] = True
        elif np.all(np.isin(job_op_1, states[2, env.state_operation_capability_location])):
            fastest_agents_available[2,0] = True

    # Check Agent 2
    if states[1, env.state_status_location_all[1]] == 0:
        # Agent 2 is available and capable
        if np.all(np.isin(job_op_2, states[1, env.state_operation_capability_location])):
            fastest_agents_available[1,1] = True
        elif np.all(np.isin(job_op_1, states[1, env.state_operation_capability_location])):
            fastest_agents_available[1,0] = True
        elif np.all(np.isin(job_op_3, states[1, env.state_operation_capability_location])):
            fastest_agents_available[1,2] = True
    
    if states[0, env.state_status_location_all[1]] == 0:
        # Agent 1 is available and capable
        if np.all(np.isin(job_op_1, states[0, env.state_operation_capability_location])):
            fastest_agents_available[0,0] = True
        elif np.all(np.isin(job_op_3, states[0, env.state_operation_capability_location])):
            fastest_agents_available[0,2] = True
        elif np.all(np.isin(job_op_2, states[0, env.state_operation_capability_location])):
            fastest_agents_available[0,1] = True



    return fastest_agents_available

def hitl_action(states, env):
    # Get the baseline mask for each agent's actions.
    mask_actions = masking_action(states, env)
    # Get FAA flags: a list where only the fastest available agent is True.
    fastest_agents_available = FAA(states, env)
    
    if fastest_agents_available[2, 0]: # Agent 3 is the fastest agent available, but the job in agent 1.
        mask_actions[0, 1] = False # Wait yr-1 in agent 1 got cancelled.
    elif fastest_agents_available[2, 1]: # Agent 3 is the fastest agent available, but the job in agent 2.
        mask_actions[1,1] = False # Wait yr-1 in agent 2 got cancelled.
    elif fastest_agents_available[2, 2]: # Agent 3 is the fastest agent available and  job in agent 3.
        pass
    
    if fastest_agents_available[1, 0]: # Agent 2 is the fastest agent available, but the job in agent 1.
        mask_actions[0, 1] = False # Wait yr-1 in agent 1 got cancelled.
    elif fastest_agents_available[1, 1]: # Agent 2 is the fastest agent available and the job in agent 2.
        pass
    elif fastest_agents_available[1, 2]: # Agent 2 is the fastest agent available and the job in agent 3.
        mask_actions[2, 1] = False # Wait yr-1 in agent 3 got cancelled.

    if fastest_agents_available[0, 0]: # Agent 1 is the fastest agent available and the job in agent 1.
        pass
    elif fastest_agents_available[0, 1]: # Agent 1 is the fastest agent available and the job in agent 2.
        mask_actions[1, 1] = False # Wait yr-1 in agent 2 got cancelled.
    elif fastest_agents_available[0, 2]: # Agent 1 is the fastest agent available and the job in agent 3.
        mask_actions[2, 1] = False # Wait yr-1 in agent 3 got cancelled.
        

    return mask_actions





if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000)
    rewards = {}
    makespan = {}
    for episode in range(1, 20+1):
        print("\nepisode:", episode)
        state, info = env.reset(seed=1000+episode)
        reward_satu_episode = 0
        done = False
        truncated = False
        #print("\nEpisode:", episode)
        #print("Initial state:", state)
        while not done and not truncated:
            if len(env.conveyor.product_completed)>= env.n_jobs:
                print("All jobs are completed.")
                break

            hitl_actions=hitl_action(state, env)
            actions=[]

            for action_agent in hitl_actions:
                true_indices = np.where(action_agent)[0]
                random_actions = true_indices[0]
                actions.append(random_actions)

            actions = np.array(actions)


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
        
        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count, "total product completed:", len(env.conveyor.product_completed))
        order = {'A': 0, 'B': 1, 'C': 2}


    env.close()
    print("rewards:", rewards)
    print("makespan:", makespan)
    file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_HITL_zz.json")
    with open(file_path, "w") as f:
        json.dump(rewards, f, indent=4)
    file_path= os.path.join(CURRENT_DIR, "Testing_makespan_HITL_zz.json")
    with open(file_path, "w") as f:
        json.dump(makespan, f, indent=4)

        #print("product sorted: ",sorted_jobs)
    print("selesai")