from env_5c_4_sekuensial import FJSPEnv
import numpy as np
import random
from tqdm import tqdm
from MASKING_ACTION_MODEL import masking_action
def FCFS_action(states):
    actions=[]
    for i, state in enumerate(states):
        if state[env.state_operation_now_location]==0:

            if state[env.state_status_location_all[i]]==1:
                actions.append(0) # accept saat di workbench
            elif (state[env.state_first_job_operation_location[0]]!=0 and 
                  state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]
                  and state[env.state_pick_job_window_location] ==1 ):
                actions.append(0) # Accept saat di conveyor yr
            elif (state[env.state_first_job_operation_location[1]]!=0 and 
                  state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location]):# ada job di yr-1
                actions.append(1) # wait yr-1
            elif (state[env.state_first_job_operation_location[2]]!=0 and 
                  state[env.state_first_job_operation_location[2]] in state[env.state_operation_capability_location]):# ada job di yr-2
                actions.append(2) # wait yr-2
            elif np.array_equal(state[env.state_first_job_operation_location], [0, 0, 0]):
                actions.append(4) # continue
            else:
                actions.append(3) # Decline

        elif state[env.state_operation_now_location]!=0:
            if state[env.state_status_location_all[i]]==2 or  state[env.state_status_location_all[i]]==3 : # agent working hingga completing
                actions.append(4) # continue
            else:
                print("FAILED ACTION: agent status is not working")
                actions.append(None)

        else:
            actions.append(None)
            print("PASTI ADA YANG SALAH")
    return actions


  

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000)
    for episode in tqdm(range(1, 1+ 10)):
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

            actions=FCFS_action(state)
            #print("actions:", actions)
            mask_actions = masking_action(state, env)
            #print("mask_actions:", mask_actions)

            for dummy, mask_action in enumerate(mask_actions):
                true_indices = np.where(mask_action)[0]
                #print("true_indices:", true_indices)
                if actions[dummy] not in true_indices:
                    print("true_indices:", true_indices)
                    print("actions:", actions)
                    actions[dummy]=None
                    print("state:", state)


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
        if  None in actions:
            break
        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count)
        order = {'A': 0, 'B': 1, 'C': 2}

        # Sorting by product type first, then by numeric value
        #print("product completed: ",env.conveyor.product_completed)
        sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

        #print("product sorted: ",sorted_jobs)
    print("selesai")