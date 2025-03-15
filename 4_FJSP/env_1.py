import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_1 import CircularConveyor
from agent_1 import Agent
'''
1. Urutan update state-> move conveyor -> generate jobs -> update state Syrt
2. sudah bisa mengembalikan barang dari workbench ke conveyor
'''

# ============================================================================
# FJSPEnv: Gymnasium Environment untuk Flexible Job Shop
# ============================================================================
class FJSPEnv(gym.Env):

    def __init__(self, window_size: int, num_agents: int, max_steps: int, episode: int):
        super(FJSPEnv, self).__init__()
        self.episode_count = episode
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
        self.reward_product_complete=0
        self.FAILED_ACTION = False

        # base processing times for each job type and operation
        self.base_processing_times ={}

        # Konfigurasi agent
        # Parameter Agent
        self.agents = []
        self.multi_agent_speeds = [[1, 2.0], [1, 2.0], [1, 2.0], [2.0, 1], [2.0, 1], [2.0, 1]] 
        self.multi_agent_energy_consumption=[0,0,0]   
        # STATE 
        # Fixed positions (indeks 0-based): Agent1: 1, Agent2: 3, Agent3: 5, Agent4: 7, Agent5: 9, Agent6: 11
        self.multi_agent_positions = [1, 3, 5, 7, 9, 11]
        self.multi_agent_operation_capability = [[1,2], [2,3], [1,3], [1,2], [2,3], [1,3]]
        #---------------------------------------------------------------------
        # STATE LOCATION
        self.state_yr_location=0
        self.state_operation_capability_location=[1,2]
        self.state_operation_now_location=3
        self.state_status_location_all = list(range(4, 4 + self.num_agents))
        self.state_workbench_remaining_operation = 4 + self.num_agents
        self.state_workbench_degree_of_completion = 5 + self.num_agents
        self.state_remaining_operation = list(range(6 + self.num_agents, 6 + self.num_agents + self.window_size))
        self.state_processing_time_remaining = list(range(6 + self.num_agents + self.window_size, 6 + self.num_agents + 2 * self.window_size))

        # print("self.state_status_location_all: ", self.state_status_location_all)
        # print("self.state_workbench_remaining_operation: ", self.state_workbench_remaining_operation)
        # print("self.state_workbench_degree_of_completion: ", self.state_workbench_degree_of_completion)
        # print("self.state_remaining_operation: ", self.state_remaining_operation)
        # print("self.state_processing_time_remaining: ", self.state_processing_time_remaining)

        #---------------------------------------------------------------------
        # Ruang observasi: tiap agen memiliki state vektor berukuran 14
        # 1. posisi conveyor (integer)
        # 2. operasi yang bisa dilakukan oleh agent  (list [o1,o2])
        # 3. operasi sekarang yang sedang dikerjakan oleh agent (integer)
        # 4. status seluruh agent (list [agent ke-1 hingga agent ke-6] dari 0=agent tidak aktif (rusak), 1=idle, 2=accept, 3=working, 4=completing)
        # 5.  remaining operation pada workbench (integer)
        # 6.  degree of completion pada workbench (float)
        # 7.  remaining operation pada window (list [o1,o2,o3])
        # 8.  processing time remaining pada window (list [t1,t2,t3])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 1+ 2+ 1+ self.num_agents+ 1+ 1+ 2*self.window_size), dtype=np.float32)
        # ---------------------------------------------------------------------
        # 3 aksi: 0=CONTINUE,  1=DECLINE, 2=ACCEPT
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
            # Increment episode counter before reinitializing
            self.episode_count += 1

            # Save the episode_count value
            current_episode_count = self.episode_count
            # Reinitialize the environment
            self.__init__(window_size=self.window_size, num_agents=self.num_agents, max_steps=self.max_steps, episode=current_episode_count)
            # Restore the episode_count value
            self.episode_count = current_episode_count
            self.step_count = 0

            # Reinitialize the conveyor and agents as needed
            self.conveyor = CircularConveyor(self.num_agents, current_episode_count=self.episode_count)
            self.base_processing_times = self.conveyor.base_processing_times
            self.agents = []
            for i in range(self.num_agents):
                agent = Agent(
                    agent_id=i+1,
                    position=self.multi_agent_positions[i],
                    operation_capability=self.multi_agent_operation_capability[i],
                    speed=self.multi_agent_speeds[i],
                    window_size=self.window_size,
                    num_agent=self.num_agents,
                )
                self.agents.append(agent)

            return self.initial_state(), {}

    
    def action_accept(self, observation, agent, i, status_location):
        pass
        return observation, agent
    

    def action_decline(self, observation, agent, i, status_location):
        '''
        DECLINE
        1. cek apakah status agent saat ini adalah idle
        2. jika ya, maka agent akan menolak job pada yr
        '''
        #print("DECLINE")
        #observation[self.state_pick_job_window_location]=0
        
        return observation, agent
    

    def action_continue(self, observation, agent, i, status_location):

        return observation, agent


    def update_state(self, observation_all,  actions):
        next_observation_all=observation_all
        # next_observation_all=np.array(next_observation_all)

        for i, agent in enumerate(self.agents):
            observation=observation_all[i]
            status_location=self.state_status_location_all[i]

            window_sections = [int(observation[self.state_yr_location]) - r for r in range(self.window_size)]
            agent.window_product=np.array(self.conveyor.conveyor)[window_sections]

            if actions[i] == 2: # ACCEPT
                observation, agent  = self.action_accept(observation, agent, i, status_location)

            elif actions[i] == 1: # DECLINE
                observation, agent  = self.action_decline(observation, agent, i, status_location)

            elif actions[i] == 0: # CONTINUE
                observation, agent  = self.action_continue(observation, agent, i, status_location)
      
        
            '''
            RETURN TO CONVEYOR
            1. cek :
                a. apakah terdapat operasi yang belum selesai pada product di workbench
                b. apakah conveyor pada yr kosong
                c. apakah action yang dipilih 4 (continue)
            2. jika ya, maka product akan dikembalikan ke conveyor
            3. jika tidak, maka status agent tetap continue dan menunggu conveyor yr kosong agar product dapat dikembalikan ke conveyor
            4. Agent akan menjadi idle dan workbench akan dikosongkan
            '''
                
            self.agents[i]=agent
            next_observation_all[i]=observation

            #--------------------------------------------------------------------------------------------
                    
        next_observation_all=np.array(next_observation_all)

        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()

        # for i, agent in enumerate(self.agents):
        #     window_sections = [int(observation_all[i][0]) - r for r in range(0, self.window_size)]
        #     agent.window_product=np.array(self.conveyor.conveyor)[window_sections]
        #     # agent.window_product=np.array(self.conveyor.conveyor)[window_sections]
        #     job_details_value= [(self.conveyor.job_details.get(self.conveyor.conveyor[job_window], [])) for job_window in window_sections]
            
        #     # remaining operation bergeser sesuai window size
        #     for j, value in enumerate(job_details_value):
        #         next_observation_all[i, self.state_first_job_operation_location[j]] = value[0] if len(value)>0 else 0
        #         next_observation_all[i, self.state_second_job_operation_location[j]] = value[1] if len(value)>1 else 0


        return next_observation_all


    def step(self, actions):
        """
        actions: array dengan panjang num_agents, masing-masing aksi dalam {0,1,2,3}
          0: ACCEPT  ambil job di yr position
          1,2: WAIT    menunggu  untuk section conveyor yr-1 hingga yr-window_size+1
          3: DECLINE menolak job di yr position dan tidak menunggu yr-1 hingga yr-window_size+1
          4: CONTINUE default jika sedang memproses/tidak ada job
        """
        self.is_action_wait_succeed=[False]*self.num_agents
        self.is_status_working_succeed=[False]*self.num_agents
        self.reward_product_complete=0
        self.step_count += 1


        next_observation_all= self.update_state(observation_all=self.observation_all, actions=actions)

        # reward_wait_all = self.reward_wait(actions, self.is_action_wait_succeed)
        # reward_working_all = self.reward_working(self.observation_all, self.is_status_working_succeed )
        # reward_step_all = self.reward_complete()
        # reward_accept_all = self.reward_accept(self.observation_all)
        #reward_agent_all=-0.75+reward_wait_all+reward_working_all+reward_step_all+reward_accept_all
        reward_agent_all=[0]*self.num_agents
        done_step = self.step_count >= self.max_steps
        truncated_step = True if len(self.conveyor.product_completed)>= self.conveyor.n_jobs else False
        self.observation_all=next_observation_all
        info_step = {"actions": actions}

        return next_observation_all, reward_agent_all, done_step, truncated_step, info_step
    
    def render(self):
        print(f"Time Step: {self.step_count}")
        print("\nNEXT STATE RENDER:")
        for a, agent in enumerate(self.agents):
            #print("self.conveyor.job_details:, ", self.conveyor.job_details)
            print(f"Status Agent {agent.id} at position {int(self.observation_all[a][0])}: {int(self.observation_all[a][self.state_status_location_all[a]]) }")
            print("window product: ", agent.window_product, "\nworkbench: ", agent.workbench)
        self.conveyor.display()

