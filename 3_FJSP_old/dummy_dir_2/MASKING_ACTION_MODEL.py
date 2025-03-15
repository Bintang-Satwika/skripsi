
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

        mask_actions.append([accept_action, wait_yr_1_action, wait_yr_2_action, decline_action, continue_action])
        #print("mask_actions:", mask_actions)
    return mask_actions
