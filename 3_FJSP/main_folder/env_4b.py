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
        self.agent_speeds = [1, 2, 0.5]  # Agent2 2x lebih cepat; Agent3 2x lebih lambat
        self.base_processing_times = [6, 4, 2]  # Contoh waktu dasar untuk tiap agen
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
        # Perbaikan di sini: kita definisikan aksi sesuai gambar:
        # A0 = ACCEPT, A1..A(w-1) = WAIT (disederhanakan jadi 1 wait),
        # Ad = DECLINE, Ac = CONTINUE
        # Sehingga total 4 aksi: 0=ACCEPT, 1=WAIT, 2=DECLINE, 3=CONTINUE
        # ---------------------------------------------------------------------
        self.action_space =  spaces.MultiDiscrete([4] * self.num_agents)  
        self.state_dim = self.observation_space.shape
        self.action_dim = 4
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
            print("\nAgent-", i, end=": ")
            observation=observation_all[i]
            yr = agent.position
            status_location=self.agent_status_location_all[i]

            window_sections = [yr - r for r in range(self.window_size)]
            window_agent_product=np.array(self.conveyor.conveyor)[window_sections]
            print("window: ",window_agent_product)
            print("agent.workbench: ", agent.workbench)
            

            if actions[i] == 0:
                '''
                ACCEPT
                A. saat job sudah di workbench
                    1. cek:
                        a. apakah operation ke-1 atau ke-2 di workbench sesuai dengan operation capability agent
                    2. state:
                        a. state status agent berubah dari accept menjadi working
                        b. state operation berubah dari 0 menjadi operation ke-1 atau ke-2 atau ke-3
                        c. state remaining operation bergser sesuai window size
                    

                B. saat job masih diconveyor yr
                    1. cek:
                        a.apakah job di conveyor di posisi agent saat ini (yr) ada atau tidak
                        b. apakah job tersebut sesuai dengan operation capability agent
                        c. apakah status agent saat ini adalah idle
                    2. jika ya, maka agent mengambil job tersebut dan memindahkan dari conveyor ke workbench
                    3. conveyor pada yr akan kosong 
                    4. State:
                        a. state status agent berubah dari idle menjadi accept
                        b. state remaining operation bergser sesuai window size
                    5. jika syarat tidak terpenuhi, maka tidak ada perubahan dari timestep sebelumnya
                '''
                if agent.workbench is not None and observation[status_location]==1:
                    observation[status_location]=2 # accept menjadi working
                    #observation[3]=agent.current_job

                    pass

                elif agent.workbench is False:
                    req_ops = self.conveyor.job_details.get(self.conveyor.conveyor[yr], [])

                    if self.conveyor.conveyor[yr] is not None and  req_ops[0] in agent.operation_capability and observation[status_location]==0:
                        print("ACCEPT")
                        print("req_ops: ", req_ops)
                        observation[status_location]=1 # idle menjadi accept
                        agent.workbench=self.conveyor.conveyor[yr]
                        self.conveyor.conveyor[yr] = None
                        # dummy= agent.process(self.conveyor.job_details)
                        # print("agent process on workbench: ", dummy)
                    else:
                        print("FAILED ACTION: agent workbench is False")

                else:
                    print("FAILED ACTION: workbench is not Empty.")
            
                   
                    # if len(req_ops) > 0:
                    #     req_op = req_ops[0]
                    #     if req_op in agent.operation_capability:
                    #         agent.current_job = agent.workbench
                    #         agent.workbench = None
                    #         agent.start_job()
                    #         observation[status_location]=2 # 


            elif actions[i] == 1:
                '''
                WAIT
                1. cek apakah status agent saat ini adalah idle
                2. jika ya, maka agent memberikan action wait untuk job pada yr-1 hingga yr-window_size+1
                3. state remaining operation berubah
                4. jika tidak, maka tidak ada perubahan dari timestep sebelumnya
                '''
                print("WAIT")
            elif actions[i] == 2:
                print("DECLINE")
            elif actions[i] == 3:
                print("CONTINUE")

                
            self.agents[i]=agent

            next_observation_all.append(observation)
                    
        
        
        
        
        next_observation_all=np.array(next_observation_all)
        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()
        for i, agent in enumerate(self.agents):
            yr = agent.position
            window_sections = [yr - r for r in range(self.window_size)]
            # window_agent_product=np.array(self.conveyor.conveyor)[window_sections]
            job_details_value= [(self.conveyor.job_details.get(self.conveyor.conveyor[job_window], [])) for job_window in window_sections]
            # remaining operation bergeser sesuai window size
            for j, value in enumerate(job_details_value):
                #print(j, value)
                next_observation_all[i, -3 + j] = len(value)

        # for idx, agent in enumerate(self.agents):
        #     print(f"Agent-{idx} workbench: {agent.workbench}")
        return next_observation_all


    def step(self, actions):
        """
        actions: array dengan panjang num_agents, masing-masing aksi dalam {0,1,2,3}
          0: ACCEPT  ambil job di yr position
          1: WAIT    menunggu  untuk section conveyor sebelum yr dan sesuai panjang window size -1
          2: DECLINE menolak job di yr position
          3: CONTINUE default jika sedang memproses/tidak ada job
        """
        self.step_count += 1
        #print("self.observation_all: ", self.observation_all)
        next_observation_all= self.update_state(observation_all=self.observation_all, actions_all=actions)
        self.observation_all=next_observation_all
        
      
        
        
        print()
        # Contoh reward: negatif dari panjang buffer.
        reward = -len(self.conveyor.buffer_jobs)
        done = self.step_count >= self.max_steps
        truncated = False
        info = {"actions": actions}
        print("next_observation_all: ", next_observation_all)
        return next_observation_all, reward, done, truncated, info

    def render(self, mode="human"):
       # print(f"Time Step: {self.step_count}")
        self.conveyor.display()
        for a, agent in enumerate(self.agents):
            #status = agent.current_job if agent.current_job is not None else "Idle"
            print(f"Status Agent {agent.id} at position {agent.position}: {int(self.observation_all[a][self.agent_status_location_all[a]]) }")
        #print("-" * 50)

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=100)
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
