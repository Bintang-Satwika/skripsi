from env_5c_3 import FJSPEnv
import numpy as np
import random
from tqdm import tqdm
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

def masking_action(states, env):
    mask_actions=[]

    for i, state in enumerate(states):
        is_agent_working=False
        is_status_idle=False
        is_status_accept=False
        is_pick_job_window_yr_1=False
        is_pick_job_window_yr_2=False
        is_job_in_capability_yr=False
        is_job_in_capability_yr_1=False
        is_job_in_capability_yr_2=False
        #[Accept, Wait yr-1, Wait yr-2, Decline, Continue]
        accept_action=False
        wait_yr_1_action=False
        wait_yr_2_action=False
        decline_action=False
        continue_action=False
        print
        if state[env.state_operation_now_location] !=0:
            is_agent_working=True
        if state[env.state_status_location_all[i]] ==0:
            is_status_idle=True
        if state[env.state_status_location_all[i]] ==1:
            is_status_accept=True
        if state[env.state_pick_job_window_location] ==1:
            is_pick_job_window_yr_1=True
        if state[env.state_pick_job_window_location]==2:
            is_pick_job_window_yr_2=True
        if (state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]):
            is_job_in_capability_yr=True
        if (state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location]):
            is_job_in_capability_yr_1=True
        if (state[env.state_first_job_operation_location[2]]in state[env.state_operation_capability_location]):
            is_job_in_capability_yr_2=True

        #[Accept, Wait yr-1, Wait yr-2, Decline, Continue]
        if is_status_accept:
            accept_action=True

        elif not is_agent_working:
            if is_status_idle and is_pick_job_window_yr_1 and is_job_in_capability_yr:
                accept_action=True
                decline_action=True
                continue_action=True

            if is_status_idle and is_job_in_capability_yr_2:
                wait_yr_2_action=True
                decline_action=True

            if is_status_idle and is_job_in_capability_yr_1:
                wait_yr_1_action=True
                decline_action=True
            
            # if not is_job_in_capability_yr and not is_job_in_capability_yr_1 and not is_job_in_capability_yr_2:
            #     continue_action=True
            continue_action=True


        elif is_agent_working:
            if not is_status_idle:
                continue_action=True

        mask_actions.append([accept_action, wait_yr_1_action, wait_yr_2_action, decline_action, continue_action])
    return mask_actions

        

       

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=600)
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

        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count, "jumlah job:", len(env.conveyor.product_completed))
        order = {'A': 0, 'B': 1, 'C': 2}

        # Sorting by product type first, then by numeric value
        #print("product completed: ",env.conveyor.product_completed)
        sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

        #print("product sorted: ",sorted_jobs)
    print("selesai")