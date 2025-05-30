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
        #[Accept, Decline, Continue]
        accept_action=False
        decline_action=False
        continue_action=False
        if state[env.state_operation_now_location] !=0:
            is_agent_working=True
        if state[env.state_status_location_all[0]] ==0:
            is_status_idle=True
        if state[env.state_status_location_all[0]] ==1:
            is_status_accept=True

        if (state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]):
            is_job_in_capability_yr=True


        #[Accept,  Decline, Continue]
        if is_status_accept:
            accept_action=True

        elif not is_agent_working:
            if is_status_idle and is_job_in_capability_yr:
                accept_action=True
                decline_action=True

            if is_status_idle and (state[env.state_first_job_operation_location[0]]==0):
                continue_action=True
            elif is_status_idle and (not is_job_in_capability_yr):
                decline_action=True

        elif is_agent_working:
            if not is_status_idle:
                continue_action=True
                decline_action=False

        mask_actions.append([accept_action,  decline_action, continue_action])
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

