from env_5c_3 import FJSPEnv
import numpy as np
import random


def masking_action(states, env):
    mask_actions=[]

    for i, state in enumerate(states):
        print("agent-", i+1)
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
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=700)
    state, info = env.reset(seed=3)
    #nv.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial state:", state)
    while not done and not truncated:
        if len(env.conveyor.product_completed)>= env.n_jobs:
            print("All jobs are completed.")
            break
        if env.FAILED_ACTION:
            print("FAILED ENV")
            break
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        #actions = env.action_space.sample()

        mask_actions=masking_action(state, env)
        actions=[]
        for state, mask_action in zip(state, mask_actions):
            print("state: ", state)
            print("mask action: ", mask_action)
            true_indices = np.where(mask_action)[0]
            random_actions = random.choice(true_indices)
            print("random actions: ", random_actions)
            
            actions.append(random_actions)


        if None in actions:
            print("FAILED ACTION: ", actions)
            break
        #print("state: ", state)
        print("Actions:", actions)
        next_state, reward, done, truncated, info = env.step(actions)
        #print("Reward:", reward)
        print("NEXT STATE:", next_state)
        total_reward += reward
        env.render()
        print()
        print("-" * 100)
        state = next_state
    print("len(env.conveyor.product_completed)", len(env.conveyor.product_completed))
    print("Episode complete. Total Reward:", total_reward, "jumlah step:", env.step_count)
    order = {'A': 0, 'B': 1, 'C': 2}

    # Sorting by product type first, then by numeric value
    sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

    print("product sorted: ",sorted_jobs)