import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_2 import CircularConveyor
from agent_2 import Agent


# ============================================================================
# FJSPEnv: Gymnasium Environment untuk Flexible Job Shop
# ============================================================================
class FJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, window_size: int, num_agents: int, max_steps: int):
        super(FJSPEnv, self).__init__()
        self.observation_all=None
        self.window_size = window_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0
        #
        self.is_action_wait_succeed=[False]*num_agents
        self.is_status_working_succeed=[False]*num_agents
        

        # Parameter Conveyor
        self.num_sections = 12
        self.max_capacity = 0.75
        self.arrival_rate = 0.4
        self.n_jobs = 20

        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, num_agents, n_jobs=self.n_jobs)

        # Konfigurasi agent
        # Fixed positions (indeks 0-based): Agent1:3, Agent2:7, Agent3:11
        self.agent_positions = [3, 3+1+self.window_size, 3+1*2+self.window_size*2]
        self.agent_operation_capability = [[1,2], [2,3], [1,3]]
        self.agent_many_operations= 2
        self.agent_speeds = [1, 2, 3]  # Agent2 2x lebih cepat; Agent3 3x lebih cepat
        self.base_processing_times = [15, 10, 6]  # Contoh waktu dasar untuk tiap agen
        self.agent_status_location_all=[4,5,6]
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                window_size=self.window_size,
                num_agent=self.num_agents
            )
            self.agents.append(agent)

        # Ruang observasi: tiap agen memiliki state vektor berukuran 1+2+1+3+3 = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 1+2+1+self.num_agents+self.window_size), dtype=np.float32)

        # ---------------------------------------------------------------------
        # A0 = ACCEPT, A1..A(w-1) = WAIT ,
        # Ad = DECLINE, Ac = CONTINUE
        # Sehingga total 4 aksi: 0=ACCEPT, 1=WAIT, 2=DECLINE, 3=CONTINUE
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
        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, self.num_agents, n_jobs=self.n_jobs)
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                window_size=self.window_size,
                num_agent=self.num_agents
            )
            self.agents.append(agent)
        return self.initial_state(), {}
    

    def update_state(self, observation_all,  actions_all):
        next_observation_all=[]
        for i, agent in enumerate(self.agents):
            print("\nAgent-", i+1, end=": ")
            observation=observation_all[i]
            yr = agent.position
            status_location=self.agent_status_location_all[i]

            window_sections = [yr - r for r in range(self.window_size)]
            window_agent_product=np.array(self.conveyor.conveyor)[window_sections]
            print("window: ",window_agent_product)
            print("agent.workbench: ", agent.workbench)

            first_operation=observation[1]
            second_operation= observation[1+1]
            current_operation=observation[1+self.agent_many_operations]
            capability_operation=[first_operation, second_operation]

            self.is_action_wait_succeed[i]=False
            self.is_status_working_succeed[i]=False
            

            if actions[i] == 0:
                '''
                ACCEPT
                A. saat job sudah di workbench
                    1. cek:
                        a. apakah urutan operation di workbench sesuai dengan operation capability agent
                    2. state:
                        a. state status agent berubah dari accept menjadi working
                        b. state operation berubah dari 0 menjadi operation ke-1 atau ke-2 atau ke-3
                        c. state remaining operation bergeser sesuai window size
                    
                B. saat job masih diconveyor yr
                    1. cek:
                        a.apakah job di conveyor di posisi agent saat ini (yr) ada atau tidak
                        b. apakah job tersebut sesuai dengan operation capability agent
                        c. apakah status agent saat ini adalah idle
                    2. jika ya, maka agent mengambil job tersebut dan memindahkan dari conveyor ke workbench
                    3. conveyor pada yr akan kosong 
                    4. State:
                        a. state status agent berubah dari idle menjadi accept
                        b. state remaining operation bergeser sesuai window size
                        c. state operation sekarang harus 0 (nol)
                    5. jika syarat tidak terpenuhi, maka tidak ada perubahan dari timestep sebelumnya
                '''
                if agent.workbench and observation[status_location]==1:
                    print("ACCEPT sudah di workbench")

                    observation[status_location]=2 # accept menjadi working
                    list_operation=list(agent.workbench.values())[0] # karena [[1,2.3]] jadi [1,2,3]
                    print("list_operation: ", list_operation)
                    # diambil operation job pertama karena sudah diurutkan dari 1 hingga 3
                    if list_operation[0] in capability_operation:
                        select_operation=int(np.where(list_operation[0]==first_operation, first_operation, second_operation))
                        print("select_operation: ", select_operation )
                        observation[1+self.agent_many_operations]=select_operation
                        
                        agent.processing_time_remaining = agent.processing_time(self.base_processing_times[select_operation-1])
                        print("agent.processing_time_remaining: ", agent.processing_time_remaining)
                    else:
                        print("FAILED ACTION: operation capability is False")
            
                elif not agent.workbench:
                    req_ops = self.conveyor.job_details.get(self.conveyor.conveyor[yr], [])

                    if self.conveyor.conveyor[yr] is not None and  req_ops[0] in capability_operation and observation[status_location]==0:
                        print("ACCEPT di conveyor yr")
                        observation[status_location]=1 # idle menjadi accept
                        agent.workbench["%s"%self.conveyor.conveyor[yr]]=req_ops
                        self.conveyor.conveyor[yr] = None
                        observation[1+self.agent_many_operations]=0
                    else:
                        print("FAILED ACTION: agent workbench is False")

                else:
                    print("FAILED ACTION: workbench is not Empty.")

            elif actions[i] == 1:
                if observation[status_location]==0:
                    '''
                    WAIT
                    1. cek apakah status agent saat ini adalah idle
                    2. jika ya, maka agent memberikan action wait untuk job pada yr-1
                    3. state remaining operation bergeser sesuai window size
                    4. jika tidak, maka tidak ada perubahan dari timestep sebelumnya
                    5. untuk menghitung reward, maka agent akan mendapatkan reward jika berhasil menunggu
                    '''
                    # product_name=self.conveyor.conveyor[yr-1]
                    # req_ops=self.conveyor.job_details.get(product_name, [])
                    print("WAIT yr-1")
                    self.is_action_wait_succeed[i]=True
                else:
                    print("FAILED ACTION: agent status is not idle")

            elif actions[i] == 2:
                if observation[status_location]==0:
                    '''
                    WAIT
                    1. cek apakah status agent saat ini adalah idle
                    2. jika ya, maka agent memberikan action wait untuk job pada yr-2
                    3. state remaining operation bergeser sesuai window size
                    4. jika tidak, maka tidak ada perubahan dari timestep sebelumnya
                    5. untuk menghitung reward, maka agent akan mendapatkan reward jika berhasil menunggu
                    '''
                    # product_name=self.conveyor.conveyor[yr-2]
                    # req_ops=self.conveyor.job_details.get(product_name, [])
                    print("WAIT yr-2")
                    self.is_action_wait_succeed[i]=True
                else:
                    print("FAILED ACTION: agent status is not idle")

            elif actions[i] == 3:
                print("DECLINE")

            elif actions[i] == 4:

                if observation[status_location]==2 and observation[1+self.agent_many_operations] !=0 and agent.workbench: # working
                    print("CONTINUE in working")
                    if agent.processing_time_remaining > 0:
                        agent.processing_time_remaining -= 1
                    elif agent.processing_time_remaining == 0:
                        observation[status_location]==3 # working menjadi completing
                        print("processing_time_remaining is 0")
                    else:
                        print("FAILED ACTION: agent.processing_time_remaining")

                elif observation[status_location]==1: 
                    print("CONTINUE in idle")
            
            self.agents[i]=agent
            next_observation_all.append(observation)

            if observation[status_location]==2:
                self.is_status_working_succeed[i]=True
            #--------------------------------------------------------------------------------------------
                    
        
        next_observation_all=np.array(next_observation_all)
        # update  state Syr,t, yakni operasi yang tersisa pada job di conveyor sesuai window size
        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()

        for i, agent in enumerate(self.agents):
            yr = agent.position
            window_sections = [yr - r for r in range(self.window_size)]
            # window_agent_product=np.array(self.conveyor.conveyor)[window_sections]
            job_details_value= [(self.conveyor.job_details.get(self.conveyor.conveyor[job_window], [])) for job_window in window_sections]
            print("job_details_value: ", job_details_value)
            # remaining operation bergeser sesuai window size
            for j, value in enumerate(job_details_value):
                print(j, value)
                next_observation_all[i, -3 + j] = len(value)

        # for idx, agent in enumerate(self.agents):
        #     print(f"Agent-{idx} workbench: {agent.workbench}")
        return next_observation_all


    def step(self, actions):
        """
        actions: array dengan panjang num_agents, masing-masing aksi dalam {0,1,2,3}
          0: ACCEPT  ambil job di yr position
          1: WAIT    menunggu  untuk section conveyor yr-1 hingga yr-window_size+1
          2: DECLINE menolak job di yr position dan tidak menunggu yr-1 hingga yr-window_size+1
          3: CONTINUE default jika sedang memproses/tidak ada job
        """
        self.step_count += 1
        next_observation_all= self.update_state(observation_all=self.observation_all, actions_all=actions)
        self
        print()

        reward_wait_all = self.reward_wait(actions, self.is_action_wait_succeed)
        reward_working_all = self.reward_working(self.observation_all, self.is_status_working_succeed ,factor_x=1)
        reward_step_all = self.reward_complete()
        reward_agent_all=-1.1+reward_wait_all+reward_working_all+reward_step_all
        # print("reward_wait_all: ", reward_wait_all)
        # print("reward_working_all: ", reward_working_all)
        # print("reward_agent_all: ", reward_agent_all)
        done_step = self.step_count >= self.max_steps
        truncated_step = True if self.conveyor.product_completed>= self.n_jobs else False
        self.observation_all=next_observation_all
        info_step = {"actions": actions}
        print("next_observation_all: ", next_observation_all)
        return next_observation_all, reward_agent_all, done_step, truncated_step, info_step
    
    def reward_wait(self, actions,  is_action_wait_succeed, k_wait=0.5):
        rewards=[]
        for i, agent in enumerate(self.agents):
            if actions[i]==1 or actions[i]==2 and is_action_wait_succeed[i]:
                rewards.append(agent.speed/sum(self.agent_speeds))
            else:
                rewards.append(0)
        return np.multiply(k_wait,rewards)


    def reward_working(self, observations, is_status_working_succeed ,factor_x, k_working=1):
        rewards=[]
        for r, agent in enumerate(self.agents):
            obs=observations[r]
            if obs[self.agent_status_location_all[r]]==2 and is_status_working_succeed[r]:
                rewards.append(agent.speed/sum(factor_x*self.agent_speeds))
            else:
                rewards.append(0)
        return np.multiply(k_working,rewards)
    
    def reward_complete(self, k_compelte=0.5):
        return 0

    def render(self, mode="human"):
       # print(f"Time Step: {self.step_count}")
        self.conveyor.display()
        for a, agent in enumerate(self.agents):
            print(f"Status Agent {agent.id} at position {agent.position}: {int(self.observation_all[a][self.agent_status_location_all[a]]) }")
        #print("-" * 50)

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=20)
    state, info = env.reset(seed=42)
    #nv.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial state:", state)
    while not done and not truncated:
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        actions = env.action_space.sample()
        actions=[0]*3
        print("state: ", state)
        print("Actions:", actions)
        next_state, reward, done, truncated, info = env.step(actions)
        print("Reward:", reward)
        total_reward += reward
        print()
        env.render()
        print("-" * 100)
        state = next_state
    print("Episode complete. Total Reward:", total_reward)
