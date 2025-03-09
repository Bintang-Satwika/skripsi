import numpy as np
import random
def MASKING_action(states, env):
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

            if is_status_idle and is_job_in_capability_yr_2:
                wait_yr_2_action=True
                decline_action=True

            if is_status_idle and is_job_in_capability_yr_1:
                wait_yr_1_action=True
                decline_action=True

            if is_status_idle and (state[env.state_first_job_operation_location[0]]==0 and state[env.state_first_job_operation_location[1]]==0 and state[env.state_first_job_operation_location[2]]==0):
                continue_action=True
            elif is_status_idle and (not is_job_in_capability_yr or not is_job_in_capability_yr_1 or not is_job_in_capability_yr_2):
                decline_action=True



        elif is_agent_working:
            if not is_status_idle:
                continue_action=True
                decline_action=False

        mask_actions.append([accept_action, wait_yr_1_action, wait_yr_2_action, decline_action, continue_action])
        #print("mask_actions:", mask_actions)
    return np.array(mask_actions)

def RANDOM_action(states, env):
    mask_actions= MASKING_action(states, env)
    actions=[]

    for single, mask_action in zip(states, mask_actions):
        true_indices = np.where(mask_action)[0]
        random_actions = random.choice(true_indices)
        #print("true_indices:", true_indices, "random_actions:", random_actions)
        actions.append(random_actions)

    actions = np.array(actions)
    return actions


def FCFS_action(states, env):
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
    return np.array(actions)


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
        a= (states[2, env.state_first_job_operation_location[0]], 
                states[2, env.state_second_job_operation_location[0]])
        b= (states[2, env.state_first_job_operation_location[1]], 
                states[2, env.state_second_job_operation_location[1]])
        c= (states[2, env.state_first_job_operation_location[2]],
                states[2, env.state_second_job_operation_location[2]])
        if (np.all(np.isin(a, states[2, env.state_operation_capability_location]))
            or np.all(np.isin(b, states[2, env.state_operation_capability_location]))
            or np.all(np.isin(c, states[2, env.state_operation_capability_location]))
            ):
            pass
        elif states[1, env.state_status_location_all[1]]  !=0 and states[0, env.state_status_location_all[1]] !=0:
            pass
        else:
            # Agent 3 is available and capable
            if np.all(np.isin(job_op_3, states[2, env.state_operation_capability_location])):
                fastest_agents_available[2,2] = True  
            elif np.all(np.isin(job_op_2, states[2, env.state_operation_capability_location])):
                fastest_agents_available[2,1] = True
            elif np.all(np.isin(job_op_1, states[2, env.state_operation_capability_location])):
                fastest_agents_available[2,0] = True

    # Check Agent 2
    if states[1, env.state_status_location_all[1]] == 0:
        a= (states[1, env.state_first_job_operation_location[0]], 
                states[1, env.state_second_job_operation_location[0]])
        b= (states[1, env.state_first_job_operation_location[1]], 
                states[1, env.state_second_job_operation_location[1]])
        c= (states[1, env.state_first_job_operation_location[2]],
                states[1, env.state_second_job_operation_location[2]])
        if (np.all(np.isin(a, states[1, env.state_operation_capability_location]))
            or np.all(np.isin(b, states[1, env.state_operation_capability_location]))
            or np.all(np.isin(c, states[1, env.state_operation_capability_location]))
            ):
            pass
        else:
            if np.all(np.isin(job_op_2, states[1, env.state_operation_capability_location])):
                fastest_agents_available[1,1] = True
            elif np.all(np.isin(job_op_1, states[1, env.state_operation_capability_location])):
                fastest_agents_available[1,0] = True
            elif np.all(np.isin(job_op_3, states[1, env.state_operation_capability_location])):
                fastest_agents_available[1,2] = True
    # check Agent 1
    if states[0, env.state_status_location_all[0]] == 0:
        a= (states[0, env.state_first_job_operation_location[0]], 
                states[0, env.state_second_job_operation_location[0]])
        b= (states[0, env.state_first_job_operation_location[1]], 
                states[0, env.state_second_job_operation_location[1]])
        c= (states[0, env.state_first_job_operation_location[2]],
                states[0, env.state_second_job_operation_location[2]])
        if (np.all(np.isin(a, states[0, env.state_operation_capability_location]))
            or np.all(np.isin(b, states[0, env.state_operation_capability_location]))
            or np.all(np.isin(c, states[0, env.state_operation_capability_location]))
            ):
            pass
        else:
            # Agent 1 is available and capable
            if np.all(np.isin(job_op_1, states[0, env.state_operation_capability_location])):
                fastest_agents_available[0,0] = True
            elif np.all(np.isin(job_op_3, states[0, env.state_operation_capability_location])):
                fastest_agents_available[0,2] = True
            elif np.all(np.isin(job_op_2, states[0, env.state_operation_capability_location])):
                fastest_agents_available[0,1] = True

    return fastest_agents_available

def HITL_action(states, env):
    # Get the baseline mask for each agent's actions.
    mask_actions = MASKING_action(states, env)
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
    
    
    actions=[]

    for action_agent in mask_actions:
        true_indices = np.where(action_agent)[0]
        random_actions = true_indices[0]
        actions.append(random_actions)

    actions = np.array(actions)
    return actions, mask_actions