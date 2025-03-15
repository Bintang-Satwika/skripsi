import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_3 import CircularConveyor
from agent_3 import Agent
'''
1. Urutan update state-> move conveyor -> generate jobs -> update state Syrt
2. sudah bisa mengembalikan barang dari workbench ke conveyor
'''

# ============================================================================
# FJSPEnv: Gymnasium Environment untuk Flexible Job Shop
# ============================================================================
class FJSPEnv(gym.Env):

    def __init__(self, window_size: int, num_agents: int, max_steps: int):
        super(FJSPEnv, self).__init__()
        self.observation_all=None
        self.window_size = window_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0
        self.is_action_wait_succeed=[False]*num_agents
        self.is_status_working_succeed=[False]*num_agents
        self.is_job_moving_to_workbench=[False]*num_agents
        self.product_return_to_conveyor=[None]*num_agents
        self.total_process_done=0
        

        # Parameter Conveyor
        self.num_sections = 12
        self.max_capacity = 0.75
        self.arrival_rate = 0.4
        self.n_jobs = 20

        #self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, num_agents, n_jobs=self.n_jobs)

        # Konfigurasi agent
        # STATE 
        # Fixed positions (indeks 0-based): Agent1:3, Agent2:7, Agent3:11
        self.agent_positions = [3, 3+1+self.window_size, 3+1*2+self.window_size*2]
        self.agent_operation_capability = [[1,2], [2,3], [1,3]]
        # self.agent_operation_now = 0
        # self.agent_status_all = [0]*num_agents
        # self.agent_first_product_operation=[0]*window_size
        # self.agent_second_product_operation=[0]*window_size
        # self.agent_pick_product_window = 1
        self.state_yr_location=0
        self.state_operation_capability_location=[1,2]
        self.state_operation_now_location=3
        self.state_status_location_all=[4,5,6]
        self.state_first_job_operation_location=[7,8,9] # yr, yr-1, yr-2
        self.state_second_job_operation_location=[10,11,12] # yr, yr-1, yr-2
        self.state_pick_job_window_location=13
        #---------------------------------------------------------------------
        self.agent_many_operations= 2
        self.agent_speeds = [1, 2, 3]  # Agent2 2x lebih cepat; Agent3 3x lebih cepat
        self.base_processing_times = [6, 10, 15]  #  waktu dasar untuk setiap operasi
        
        self.agents = []

        self.FAILED_ACTION = False


        # Ruang observasi: tiap agen memiliki state vektor berukuran 14
        # 1. posisi conveyor (integer)
        # 2. operasi yang bisa dilakukan oleh agent  (list [o1,o2])
        # 3. operasi sekarang yang sedang dikerjakan oleh agent (integer)
        # 4. status seluruh agent (list [s1,s2,s3] dari 0=idle, 1=accept, 2=working, 3=completing)
        # 5. operasi pertama job di conveyor (list [yr,yr-1,yr-2] dari 0 hingga 3, 0=tidak ada job, 1= operasi ke-1, 2=operasi ke-2, 3=operasi ke-3)
        # 6. operasi kedua job di conveyor (list[yr,yr-1,y-2])
        # 7. salah satu window product pada conveyor (yr-1 atau yr-2) yang dipilih oleh action Waiit 


        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 1+2+1+self.num_agents+2*self.window_size+1), dtype=np.float32)
        print("Observation Space:", self.observation_space)

        # ---------------------------------------------------------------------
        # A0 = ACCEPT, A1..A(w-1) = WAIT ,
        # Ad = DECLINE, Ac = CONTINUE
        # Sehingga total 5 aksi: 0=ACCEPT, 1=WAIT yr-1. 2=WAit yr-2, 3=DECLINE, 4=CONTINUE
        # ---------------------------------------------------------------------
        self.action_space =  spaces.MultiDiscrete([3+self.window_size-1] * self.num_agents)  
        self.state_dim = self.observation_space.shape
        self.action_dim = 3+self.window_size-1
        print(f"Dimensi State: {self.state_dim}, Dimensi Aksi: {self.action_dim}")

    def initial_state(self):
        # Update state masing-masing agen dengan memanggil build_state()
        obs = []
        for agent in self.agents:
            obs.append(agent.build_state())
        self.observation_all=np.array(obs)
        return self.observation_all
     

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # Reset conveyor dan agen
        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, 
                                         self.num_agents, n_jobs=self.n_jobs)
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                window_size=self.window_size,
                num_agent=self.num_agents,
            )
            self.agents.append(agent)
        return self.initial_state(), {}

        
    def action_accept(self, observation, agent, i, status_location):
        """
        Revised ACCEPT:
        - If the agent is idle (operation_now == 0) and has an empty workbench:
            A. If the agent is already in “accept” status (status==1) and a job is in the buffer 
                (is_job_moving_to_workbench==True), then move the buffered job to the workbench,
                update status to working (2), set operation based on the first required operation.
            B. Else if the agent is idle (status==0) and the pick window indicator is 1,
                then pick up the job from the conveyor at the current position (yr) if it matches
                the agent’s capabilities. Remove that job from the conveyor.
        """
        if observation[self.state_operation_now_location] == 0 and not agent.workbench:
            # (A) Job already buffered (waiting to be moved to workbench)
            if observation[status_location] == 1 and self.is_job_moving_to_workbench[i]:
                print("ACCEPT: job already buffered in workbench")
                self.is_job_moving_to_workbench[i] = False
                agent.workbench = agent.buffer_job_to_workbench
                agent.buffer_job_to_workbench = {}
                observation[self.state_pick_job_window_location] = 0

                observation[status_location] = 2  # Change status from accept to working

                # Get the list of operations required for the job.
                list_operation = list(agent.workbench.values())[0]  # e.g., [1,2,3]
                print("list_operation:", list_operation)

                # Select the first operation if it is within the agent’s capability.
                if list_operation[0] in observation[self.state_operation_capability_location]:
                    # Simple selection: if first element matches, choose it; otherwise choose the other.
                    if list_operation[0] == observation[self.state_operation_capability_location][0]:
                        select_operation = observation[self.state_operation_capability_location][0]
                    else:
                        select_operation = observation[self.state_operation_capability_location][1]
                    observation[self.state_operation_now_location] = select_operation
                    # Clear the job operations from the state for the position where the job was taken.
                    observation[self.state_first_job_operation_location[0]] = 0
                    observation[self.state_second_job_operation_location[0]] = 0

                    # Start processing time based on the selected operation.
                    agent.processing_time_remaining = agent.processing_time(self.base_processing_times[select_operation - 1])
                    print("agent.processing_time_remaining:", agent.processing_time_remaining)
                else:
                    print("FAILED ACTION: operation capability mismatch")
                    self.FAILED_ACTION = True
            # (B) Job is still on the conveyor.
            elif observation[status_location] == 0 and observation[self.state_pick_job_window_location] == 1 and not self.is_job_moving_to_workbench[i]:
                # Check if the job at current position (yr) is within capability.
                if observation[self.state_first_job_operation_location[0]] in observation[self.state_operation_capability_location]:
                    print("ACCEPT: picking job from conveyor at yr")
                    observation[status_location] = 1  # Change status from idle to accept.
                    self.is_job_moving_to_workbench[i] = True
                    # Buffer the job from the conveyor into the agent’s workbench buffer.
                    key = "%s" % agent.window_product[0]
                    agent.buffer_job_to_workbench[key] = self.conveyor.job_details.get(agent.window_product[0], [])
                    # Remove the job from the conveyor.
                    self.conveyor.conveyor[int(observation[self.state_yr_location])] = None
                    print("Updated conveyor:", self.conveyor.conveyor)
                else:
                    print("FAILED ACTION: job at yr not in agent's capability")
                    self.FAILED_ACTION = True
            else:
                print("FAILED ACTION: agent not idle or no job in conveyor")
                self.FAILED_ACTION = True
        else:
            print("FAILED ACTION: workbench is not empty.")
            self.FAILED_ACTION = True
        return observation, agent


    def action_wait(self, observation, agent, i, status_location, actions_all):
        """
        Revised WAIT:
        - When the agent is idle (status==0), check waiting positions.
        - If the job at position yr-1 is available and action==1, set the pick window indicator.
        - Else if the job at yr-2 is available and action==2, do the same.
        """
        if observation[status_location] == 0:
            if observation[self.state_first_job_operation_location[1]] != 0 and actions_all[i] == 1:
                print("WAIT: waiting for job at yr-1")
                observation[self.state_pick_job_window_location] = 1
                self.is_action_wait_succeed[i] = True
            elif observation[self.state_first_job_operation_location[2]] != 0 and actions_all[i] == 2:
                print("WAIT: waiting for job at yr-2")
                observation[self.state_pick_job_window_location] = 2
                self.is_action_wait_succeed[i] = True
            else:
                print("FAILED ACTION: no job available in waiting positions")
                self.FAILED_ACTION = True
        else:
            print("FAILED ACTION: agent status is not idle")
            self.FAILED_ACTION = True
        return observation, agent


    def action_decline(self, observation, agent, i, status_location):
        """
        Revised DECLINE simply resets the pick job window indicator.
        """
        print("DECLINE action")
        observation[self.state_pick_job_window_location] = 0
        return observation, agent


    def action_continue(self, observation, agent, i, status_location):
        """
        Revised CONTINUE:
        - If the agent is in working status (status==2) and has a workbench, decrement processing time.
        - When processing time reaches 0, change status to completing (3).
        - For idle agents, simply do nothing.
        """
        if observation[status_location] == 2 and agent.workbench:
            print("CONTINUE: agent working")
            self.is_status_working_succeed[i] = True
            if agent.processing_time_remaining > 0:
                agent.processing_time_remaining -= 1
                print("agent.processing_time_remaining:", agent.processing_time_remaining)
                self.total_process_done += 1
            elif agent.processing_time_remaining == 0:
                observation[status_location] = 3  # working -> completing
                self.is_status_working_succeed[i] = False
                self.total_process_done += 1
                print("Processing complete; status now:", observation[status_location])
            else:
                print("FAILED ACTION: invalid processing_time_remaining")
                self.FAILED_ACTION = True
        elif observation[status_location] == 1:
            print("CONTINUE: agent idle")
        return observation, agent


    def update_state(self, observation_all, actions_all):
        """
        Revised update_state:
        - Work on a copy of the observations.
        - Use the correct variable name (actions_all[i] instead of actions[i]).
        - Update each agent’s state based on the chosen action.
        - Then, update the conveyor (move, generate jobs) and refresh the window job operations.
        """
        # Make a copy of observation_all to avoid in-place modification issues.
        next_observation_all = observation_all.copy()

        for i, agent in enumerate(self.agents):
            print("\nAgent-", i + 1, end=": ")
            observation = next_observation_all[i]
            status_location = self.state_status_location_all[i]

            # Update the agent’s window based on its current position.
            window_sections = [int(observation[self.state_yr_location]) - r for r in range(self.window_size)]
            agent.window_product = np.array(self.conveyor.conveyor)[window_sections]

            # Reset wait/working flags.
            self.is_action_wait_succeed[i] = False
            self.is_status_working_succeed[i] = False

            # Use actions_all (not a misnamed variable) for decision branches.
            if actions_all[i] == 0:  # ACCEPT
                observation, agent = self.action_accept(observation, agent, i, status_location)
            elif actions_all[i] == 1 or actions_all[i] == 2:  # WAIT
                observation, agent = self.action_wait(observation, agent, i, status_location, actions_all)
            elif actions_all[i] == 3:  # DECLINE
                observation, agent = self.action_decline(observation, agent, i, status_location)
            elif actions_all[i] == 4:  # CONTINUE
                observation, agent = self.action_continue(observation, agent, i, status_location)

            # RETURN TO CONVEYOR (if needed)
            if self.product_return_to_conveyor[i] and agent.workbench:
                print("RETURN TO CONVEYOR: attempting to return product")
                current_pos = int(observation[self.state_yr_location])
                if self.conveyor.conveyor[current_pos] is None:
                    # Return the job to the conveyor.
                    key = list(agent.workbench.keys())[0]
                    self.conveyor.conveyor[current_pos] = str(key)
                    print("Product returned to conveyor:", self.conveyor.conveyor)
                    self.product_return_to_conveyor[i] = False
                    agent.workbench = {}
                    observation[status_location] = 0
                    # Reset the agent’s operation (index 3)
                    observation[self.state_operation_now_location] = 0
                else:
                    print("RETURN FAILED: conveyor position not empty")

            # COMPLETING processing: if status is completing (3) and no pending return
            if observation[status_location] == 3 and not self.product_return_to_conveyor[i]:
                print("Completing job, agent.workbench before:", agent.workbench)
                for key, ops in list(agent.workbench.items()):
                    if len(ops) > 1:
                        agent.workbench[key].pop(0)
                    else:
                        self.conveyor.product_completed.append(key)
                        agent.workbench = {}
                        observation[status_location] = 0
                        observation[self.state_operation_now_location] = 0
                    # If job still remains, decide whether to return it.
                    if agent.workbench:
                        if list(agent.workbench.values())[0][0] in observation[self.state_operation_capability_location]:
                            print("Job can continue processing")
                            self.product_return_to_conveyor[i] = False
                            observation[status_location] = 2
                            observation[self.state_operation_now_location] = list(agent.workbench.values())[0][0]
                        else:
                            self.product_return_to_conveyor[i] = True
                    # Update job details.
                    try:
                        self.conveyor.job_details[key] = agent.workbench[key]
                    except:
                        self.conveyor.job_details.pop(key)
                print("Completing job, agent.workbench after:", agent.workbench)

            self.agents[i] = agent
            next_observation_all[i] = observation

        # Convert to numpy array (if needed)
        next_observation_all = np.array(next_observation_all)

        # Move the conveyor and generate new jobs.
        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()

        # Update the job operation details in the observation for each agent.
        for i, agent in enumerate(self.agents):
            window_sections = [int(next_observation_all[i][0]) - r for r in range(self.window_size)]
            agent.window_product = np.array(self.conveyor.conveyor)[window_sections]
            job_details_value = [self.conveyor.job_details.get(self.conveyor.conveyor[job_window], []) for job_window in window_sections]
            for j, value in enumerate(job_details_value):
                next_observation_all[i, self.state_first_job_operation_location[j]] = value[0] if len(value) > 0 else 0
                next_observation_all[i, self.state_second_job_operation_location[j]] = value[1] if len(value) > 1 else 0

        return next_observation_all

    def step(self, actions):
        """
        actions: array dengan panjang num_agents, masing-masing aksi dalam {0,1,2,3}
          0: ACCEPT  ambil job di yr position
          1,2: WAIT    menunggu  untuk section conveyor yr-1 hingga yr-window_size+1
          3: DECLINE menolak job di yr position dan tidak menunggu yr-1 hingga yr-window_size+1
          4: CONTINUE default jika sedang memproses/tidak ada job
        """
        
        self.step_count += 1


        next_observation_all= self.update_state(observation_all=self.observation_all, actions_all=actions)
        print()

        reward_wait_all = self.reward_wait(actions, self.is_action_wait_succeed,factor_x=1)
        reward_working_all = self.reward_working(self.observation_all, self.is_status_working_succeed )
        reward_step_all = self.reward_complete()
        reward_agent_all=-1.1+reward_wait_all+reward_working_all+reward_step_all
        # print("reward_wait_all: ", reward_wait_all)
        # print("reward_working_all: ", reward_working_all)
        # print("reward_agent_all: ", reward_agent_all)
        done_step = self.step_count >= self.max_steps
        truncated_step = True if len(self.conveyor.product_completed)>= self.n_jobs else False
        self.observation_all=next_observation_all
        info_step = {"actions": actions}

        return next_observation_all, reward_agent_all, done_step, truncated_step, info_step
    
    def reward_wait(self, actions,  is_action_wait_succeed, factor_x, k_wait=0.5):
        rewards=[]
        for i, agent in enumerate(self.agents):
            if actions[i]==1 or actions[i]==2 and is_action_wait_succeed[i]:
                rewards.append(agent.speed/sum(factor_x*self.agent_speeds))
            else:
                rewards.append(0)
        return np.multiply(k_wait,rewards)


    def reward_working(self, observations, is_status_working_succeed , k_working=1):
        rewards=[]
        for r, agent in enumerate(self.agents):
            obs=observations[r]
            if obs[self.state_status_location_all[r]]==2 and is_status_working_succeed[r]:
                rewards.append(agent.speed/sum(self.agent_speeds))
            else:
                rewards.append(0)
        return np.multiply(k_working,rewards)
    
    def reward_complete(self, k_complete=0.5):
        value = k_complete*self.total_process_done
        self.total_process_done=0
        return value

    def render(self):
       # print(f"Time Step: {self.step_count}")
        print("\nNEXT STATE RENDER:")
        for a, agent in enumerate(self.agents):
            #print("self.conveyor.job_details:, ", self.conveyor.job_details)
            print(f"Status Agent {agent.id} at position {int(self.observation_all[a][0])}: {int(self.observation_all[a][self.state_status_location_all[a]]) }")
            print("window product: ", agent.window_product, "\nworkbench: ", agent.workbench)
            print()
        self.conveyor.display()
        #print("-" * 50)


def FCFS_action(states):
    actions=[]
    for i, state in enumerate(states):
        if state[env.state_operation_now_location]==0:

            if state[env.state_status_location_all[i]]==1:
                actions.append(0) # accept saat di workbench
            elif (state[env.state_first_job_operation_location[0]]!=0 and 
                  state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]): # ada job di yr
                actions.append(0) # Accept saat di conveyor
            elif (state[env.state_first_job_operation_location[1]]!=0 and 
                  state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location]):# ada job di yr-1
                actions.append(1) # wait yr-1
            elif (state[env.state_first_job_operation_location[2]]!=0 and 
                  state[env.state_first_job_operation_location[2]] in state[env.state_operation_capability_location]):# ada job di yr-2
                actions.append(2) # wait yr-2
            elif np.array_equal(state[env.state_first_job_operation_location], [0, 0, 0]):
                #print("i:", i)
                actions.append(4) # continue
            else:
                print("milih decline")
                actions.append(3) # Decline

        elif state[env.state_operation_now_location]!=0 and state[env.state_status_location_all[i]]==2: # agent working hingga completing
                actions.append(4) # continue

        else:
            actions.append(None)
    return actions

def FAA_action(states):
    """
    FAA_action: Fastest Available Agent action selection.
    
    Processes agents in the order [2, 1, 0] (agent3 fastest, then agent2, then agent1).
    For each agent (using only its state), if the agent is idle 
    (i.e. state[env.state_operation_now_location] == 0), it examines its job window.
    
    The window is defined as:
      - Current position (yr) from index 0,
      - Followed by positions yr-1 and yr-2.
      
    The job operations for these positions are stored at indices defined by
    env.state_first_job_operation_location (i.e. indices [7, 8, 9]).
    
    The agent's capabilities are given by the values at indices defined in
    env.state_operation_capability_location (i.e. indices [1, 2]).
    
    The decision rules are:
      - If a job at the current position is nonzero, is within the agent’s capability,
        and hasn’t already been claimed by a faster agent, then choose ACCEPT (action 0).
      - Otherwise, check the next positions in order (action 1 for yr-1, action 2 for yr-2).
      - If the entire window is empty, choose CONTINUE (action 4).
      - Otherwise, choose DECLINE (action 3).
      
    For agents that are already processing a job (operation_now ≠ 0),
    the action is set to CONTINUE (action 4).
    
    Note: This implementation uses the global `env` to access the state index mappings.
    """
    actions = [None] * len(states)
    # Define processing order: fastest (agent3: index 2) to slowest (agent1: index 0)
    fastest_order = [2, 1, 0]
    # Dictionary to track claimed conveyor positions so that a slower agent doesn't choose a job already taken.
    claimed_positions = {}

    for i in fastest_order:
        state = states[i]
        # Check if the agent is idle.
        if state[env.state_operation_now_location] != 0:
            actions[i] = 4  # If busy, simply CONTINUE.
            continue

        # Determine the agent's current position (yr) from index 0.
        current_pos = int(state[env.state_yr_location])
        # Define window positions: [yr, yr-1, yr-2]
        window_positions = [current_pos - r for r in range(3)]
        # Extract job operations from the first job operation indices (7, 8, 9).
        job_ops = [state[idx] for idx in env.state_first_job_operation_location]
        # Extract the agent's operation capabilities from indices [1, 2].
        capabilities = [state[idx] for idx in env.state_operation_capability_location]

        # Decision logic based on the window:
        if job_ops[0] != 0 and (job_ops[0] in capabilities) and (window_positions[0] not in claimed_positions):
            actions[i] = 0  # ACCEPT job at current position.
            claimed_positions[window_positions[0]] = True
        elif job_ops[1] != 0 and (job_ops[1] in capabilities) and (window_positions[1] not in claimed_positions):
            actions[i] = 1  # WAIT for job at yr-1.
            claimed_positions[window_positions[1]] = True
        elif job_ops[2] != 0 and (job_ops[2] in capabilities) and (window_positions[2] not in claimed_positions):
            actions[i] = 2  # WAIT for job at yr-2.
            claimed_positions[window_positions[2]] = True
        elif all(op == 0 for op in job_ops):
            actions[i] = 4  # CONTINUE if the window is completely empty.
        else:
            actions[i] = 3  # DECLINE as a fallback.
    
    return actions


if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=500)
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
            print("Failed action detected. Exiting.")
            break
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        #actions = env.action_space.sample()

        #actions=FCFS_action(state)
        actions=FAA_action(state)
        #actions=[0]*3
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
    print("even.n_jobs", env.n_jobs)
    print("Episode complete. Total Reward:", total_reward, "jumlah step:", env.step_count)
    # Define sorting order for product types
    order = {'A': 0, 'B': 1, 'C': 2}

    # Sorting by product type first, then by numeric value
    sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

    print("compelted product sorted: ",sorted_jobs)
