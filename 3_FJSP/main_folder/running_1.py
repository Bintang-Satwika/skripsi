from env_5c import FJSPEnv
import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
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
                print("milih decline")
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
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=200)
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

        actions=FCFS_action(state)
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