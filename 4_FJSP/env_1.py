import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_4 import CircularConveyor
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
        self.multi_agent_speeds = [[1, 2.0], [1, 2.0], [1, 2.0], [2.0, 1], [2.0, 1], [2.0, 1]] 
        self.multi_agent = []
        self.multi_agent_energy_consumption=[0,0,0]   

        # STATE 
        # Fixed positions (indeks 0-based): Agent1: 1, Agent2: 3, Agent3: 5, Agent4: 7, Agent5: 9, Agent6: 11
        self.multi_agent_positions = [1, 3, 5, 7, 9, 11]
        self.mult_agent_operation_capability = [[1,2], [2,3], [1,3], [1,2], [2,3], [1,3]]
        #---------------------------------------------------------------------
        # STATE LOCATION
        self.state_yr_location=0
        self.state_operation_capability_location=[1,2]
        self.state_operation_now_location=3
        # self.state_status_location_all=[4,5,6,7,8,9]
        # self.state_workbench_remaining_operation=10
        # self.state_workbench_degree_of_completion=11
        # self.state_remaining_operation=[12,13] # window size=2
        # self.state_processing_time_remaining=[14,15] # window size=2
        self.state_status_location_all = list(range(4, 4 + self.num_agents))
        self.state_workbench_remaining_operation = 4 + self.num_agents
        self.state_workbench_degree_of_completion = 5 + self.num_agents
        self.state_remaining_operation = list(range(6 + self.num_agents, 6 + self.num_agents + self.window_size))
        self.state_processing_time_remaining = list(range(6 + self.num_agents + self.window_size, 6 + self.num_agents + 2 * self.window_size))

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
            self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, 
                                            self.num_agents, n_jobs=self.n_jobs, current_episode_count=self.episode_count)
            self.base_processing_times = self.conveyor.base_processing_times
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
        '''
        ACCEPT
        A. saat job sudah di workbench
            1. cek syarat:
                a. apakah  status agent sudah accept
                b. apakah job sedang dipindahkan ke workbench
            2. jika ya, maka job sudah ada di workbench dan agent akan memproses job tersebut
            3. state:
                a. state status agent berubah dari accept menjadi working
                b. state operation berubah dari 0 menjadi operation ke-1 atau ke-2 atau ke-3
                c. state remaining operation bergeser sesuai window size
            4. jika syarat tidak terpenuhi, maka job masih diconveyor yr 
            
        B. saat job masih diconveyor yr
            1. cek:
                a.apakah job di conveyor di posisi agent saat ini (yr) ada atau tidak
                b. apakah job tersebut sesuai dengan operation capability agent
                c. apakah status agent saat ini adalah idle
            2. jika ya, maka agent mengambil job tersebut dan memindahkan ke buffer "is_job_moving_to_workbench"
            3. conveyor pada yr akan kosong 
            4. State:
                a. state status agent berubah dari idle menjadi accept
                b. state remaining operation bergeser sesuai window size
                c. state operation sekarang harus 0 (nol)
            5. jika syarat tidak terpenuhi, maka tidak ada perubahan dari timestep sebelumnya
        '''
        if observation[self.state_operation_now_location]==0  and not agent.workbench:

            if  observation[status_location]==1 and self.is_job_moving_to_workbench[i] :
                #print("ACCEPT sudah di workbench")
                self.is_job_moving_to_workbench[i]=False
                agent.workbench=agent.buffer_job_to_workbench
                agent.buffer_job_to_workbench={}
                observation[self.state_pick_job_window_location]=0

                observation[status_location]=2 # accept menjadi working

                list_operation=list(agent.workbench.values())[0] # karena [[1,2.3]] jadi [1,2,3]
                # diambil operation job pertama karena sudah diurutkan dari 1 hingga 3
                if list_operation[0] in observation[self.state_operation_capability_location]:
                    select_operation=int(np.where(list_operation[0]==observation[self.state_operation_capability_location][0], 
                                                    observation[self.state_operation_capability_location][0], 
                                                    observation[self.state_operation_capability_location][1])
                                                )
                    #  state operation berubah dari 0 menjadi operation ke-1 atau ke-2 atau ke-3
                    observation[self.state_operation_now_location] =select_operation
                    # state first job operation akan menjadi nol pada yr karena job dipindah ke workbench
                    observation[self.state_first_job_operation_location[0]]=0
                    # state second job operation akan menjadi nol pada yr karena job  dipindah ke workbench
                    observation[self.state_second_job_operation_location[0]]=0
                    
                    dummy = list(agent.workbench.keys())[0].split("-")[0]
                    # print(" self.base_processing_times[dummy]", self.base_processing_times[dummy])
                    # print("select_operation-1", select_operation-1)
                    # print("self.base_processing_times[dummy][select_operation-1]", self.base_processing_times[dummy][select_operation-1])
                    # processing time agent akan mulai dihitung
                    agent.processing_time_remaining =  agent.processing_time( self.base_processing_times[dummy][select_operation-1])
                    #print("agent.processing_time_remaining: ", agent.processing_time_remaining)
                else:
                    print("FAILED ACTION: operation capability is False")
                    print("agent-%d"%i)
                    self.FAILED_ACTION=True
        
            elif observation[status_location]==0  and observation[self.state_pick_job_window_location]== 1 and not self.is_job_moving_to_workbench[i] :
                if  observation[self.state_first_job_operation_location[0]] in  observation[self.state_operation_capability_location]:
                    #print("ACCEPT di conveyor yr")
                    # idle menjadi accept
                    observation[status_location]=1 
                    # job akan dipindahkan ke workbench
                    self.is_job_moving_to_workbench[i]=True
                    # menyimpan job conveyor yr di buffer untuk dipindahkan ke workbench pada timestep selanjutnya
                    agent.buffer_job_to_workbench["%s"%agent.window_product[0]]=  self.conveyor.job_details.get(agent.window_product[0], [])
                    # conveyor pada yr  akan kosong
                    self.is_job_conveyor_yr_remove=True
                    self.conveyor.conveyor[int(observation[self.state_yr_location])] = None
                    #print("self.conveyor.conveyor: ", self.conveyor.conveyor)
                else:
                    print("FAILED ACTION: observation first job operation is not in operation capability")
                    self.FAILED_ACTION=True
            else:
                print("FAILED ACTION: agent status is not idle or job is not in conveyor")
                print("XXXXXXXXXXXXXXXXXXX")
                print("agent-%d"%i)
                self.FAILED_ACTION=True

        else:
            print("FAILED ACTION: workbench is not Empty.")
            self.FAILED_ACTION=True
        
        return observation, agent
    

    def action_wait(self, observation, agent, i, status_location, actions):
        if observation[status_location]==0:
            
            if observation[self.state_first_job_operation_location[1]] !=0 and actions[i]==1:
                '''
                WAIT
                1. cek apakah status agent saat ini adalah idle
                2. jika ya, maka agent memberikan action wait untuk job pada yr-1
                3. state remaining operation bergeser sesuai window size
                4. jika tidak, maka tidak ada perubahan dari timestep sebelumnya
                5. untuk menghitung reward, maka agent akan mendapatkan reward jika berhasil menunggu
                '''
                #print("WAIT yr-1")
                observation[self.state_pick_job_window_location]=1
                self.is_action_wait_succeed[i]=True
                

            elif observation[self.state_first_job_operation_location[2]] !=0 and actions[i]==2:
                '''
                WAIT
                1. cek apakah status agent saat ini adalah idle
                2. jika ya, maka agent memberikan action wait untuk job pada yr-2
                3. state remaining operation bergeser sesuai window size
                4. jika tidak, maka tidak ada perubahan dari timestep sebelumnya
                5. untuk menghitung reward, maka agent akan mendapatkan reward jika berhasil menunggu
                '''
                #print("WAIT yr-2")
                observation[self.state_pick_job_window_location]=2
                self.is_action_wait_succeed[i]=True
                
            else:
                print("FAILED ACTION: there is no any job in conveyor yr-2 or yr-1")
                self.FAILED_ACTION=True
                
        else:
            print("FAILED ACTION: agent status is not idle")
            self.FAILED_ACTION=True

        return observation, agent
    

    def action_decline(self, observation, agent, i, status_location):
        '''
        DECLINE
        1. cek apakah status agent saat ini adalah idle
        2. jika ya, maka agent akan menolak job pada yr
        '''
        #print("DECLINE")
        observation[self.state_pick_job_window_location]=0
        
        return observation, agent
    

    def action_continue(self, observation, agent, i, status_location):
        '''
        CONTINUE
        A. saat agent sedang bekerja di workbench
        1. cek apakah status agent saat ini adalah working pada workbench
        2. jika ya, maka waktu process akan berkurang 1 iterasi
        3. jika waktu process sudah 0, maka status agent akan berubah dari working menjadi completing
        4. State:
            a. state status agent berubah dari working menjadi completing
            b. state operation berubah  menjadi 0
            c. state remaining operation bergeser sesuai window size

        B. saat agent idle
        1. cek apakah status agent saat ini adalah idle
        2. jika ya, maka tidak ada perubahan dari timestep sebelumnya dan action wait akan diberikan pada conveyor yang sama.
        '''

        if observation[status_location]==2 and observation[1+self.agent_many_operations] !=0 and agent.workbench: # sekarang lagi working di workbench
            #print("CONTINUE in working")
            # agent sedang bekerja
            self.is_status_working_succeed[i]=True
            if agent.processing_time_remaining > 0:
                agent.processing_time_remaining -= 1
                self.agents_energy_consumption[i]+=1 # energy consumption
            elif agent.processing_time_remaining == 0:
                observation[status_location]= 3 # working menjadi completing
                self.is_status_working_succeed[i]=False # operasi selesai dan agent tidak bekerja
                self.total_process_done+=1
                #print("processing_time_remaining is 0")

            else:
                print("FAILED ACTION: agent.processing_time_remaining")
                self.FAILED_ACTION=True

        elif observation[status_location]==1: 
            #print("CONTINUE in idle")
            pass
            

        return observation, agent


    def update_state(self, observation_all,  actions):
        next_observation_all=observation_all
        # next_observation_all=np.array(next_observation_all)

        for i, agent in enumerate(self.agents):
            #print("\nAgent-", i+1, end=": ")
            #print("window product: ", agent.window_product, "\nworkbench: ", agent.workbench)
            observation=observation_all[i]
            status_location=self.state_status_location_all[i]
            #print("int(observation[self.state_yr_location])", int(observation[self.state_yr_location]))

            window_sections = [int(observation[self.state_yr_location]) - r for r in range(self.window_size)]
            #print("window_sections: ", window_sections)
            agent.window_product=np.array(self.conveyor.conveyor)[window_sections]


            #capability_operation=[obs_first_operation, obs_second_operation]

            self.is_action_wait_succeed[i]=False
            self.is_status_working_succeed[i]=False
            

            if actions[i] == 0: # ACCEPT
                observation, agent  = self.action_accept(observation, agent, i, status_location)
            
            elif actions[i] == 1 or actions[i] == 2: # WAIT
                observation, agent  = self.action_wait(observation, agent, i, status_location, actions)

            elif actions[i] == 3: # DECLINE
                observation, agent  = self.action_decline(observation, agent, i, status_location)

            elif actions[i] == 4: # CONTINUE
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
            # mengembalikan product ke conveyor jika operasi belum selesai dan product masih ada operasi selanjutnya
            # (kodingan ini harus ditaruh sebelum self.product_return_to_conveyor[i] menjadi True)
            if self.product_return_to_conveyor[i] and agent.workbench:
                    # jika conveyor pada yr kosong, maka product akan dikembalikan ke conveyor
                    if self.conveyor.conveyor[int(observation[self.state_yr_location])] is None:
                        # mengembalikan product ke conveyor
                        self.conveyor.conveyor[int(observation[self.state_yr_location])]=str(list(agent.workbench)[0])
                        #print("self.conveyor.conveyor: ", self.conveyor.conveyor)
                        self.product_return_to_conveyor[i]= False
                        # mengosongkan workbench -> agent menjadi idle -> tidak ada operasi pada robot
                        agent.workbench={}
                        observation[status_location]=0
                        observation[1+self.agent_many_operations]=0
                    else:
                        #print("CAN'T RETURN: conveyor yr is not empty")
                        pass
            
            if observation[status_location]==3 and not self.product_return_to_conveyor[i]:
                #print("COMPLETING masuk sini")
                #print("agent.workbench before: ", agent.workbench)
                for key, vektor in agent.workbench.items():
                    # mengurangi operation yang sudah selesai
                    if len(vektor)>1:
                        agent.workbench[key].pop(0)
                    else:
                        # menghapus product yang sudah selesai dan mengosongkan workbench
                        self.conveyor.product_completed.append(key)
                        agent.workbench={}
                        observation[status_location]=0
                        observation[self.state_operation_now_location]=0
                        #self.reward_product_complete+=1
                        #print("self.conveyor.job_details: ", self.conveyor.job_details)
                        #print("agent.workbench after: ", agent.workbench)
                    if agent.workbench:
                        if list(agent.workbench.values())[0][0] in observation[self.state_operation_capability_location]:
                            #print( list(agent.workbench.values())[0][0] )
                            self.product_return_to_conveyor[i]=False
                            observation[status_location]=2
                            observation[self.state_operation_now_location]=list(agent.workbench.values())[0][0]
                        else:
                            # mengembalikan ke conveyor jika operasi product belum selesai
                            self.product_return_to_conveyor[i]=True
                        
                    # mengupdate job_details
                    try:
                        self.conveyor.job_details[key]= agent.workbench[key]
                    except:
                        self.conveyor.job_details.pop(key)
                
                    #print("self.conveyor.job_details: ", self.conveyor.job_details)
                #print("agent.workbench after: ", agent.workbench)

                # menyimpan job ke buffer untuk iterasi selanjutnya agar dapat dipindahkan ke conveyor
                #agent.buffer_job_to_conveyor=agent.workbench
                
            # bila tidak ada operasi job sama sekalipun
            if sum(observation[self.state_first_job_operation_location])==0:
                observation[self.state_pick_job_window_location]=0
                
            self.agents[i]=agent
            next_observation_all[i]=observation

            #--------------------------------------------------------------------------------------------
                    
        
        next_observation_all=np.array(next_observation_all)

        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()

        for i, agent in enumerate(self.agents):
            window_sections = [int(observation_all[i][0]) - r for r in range(0, self.window_size)]
            agent.window_product=np.array(self.conveyor.conveyor)[window_sections]
            # agent.window_product=np.array(self.conveyor.conveyor)[window_sections]
            job_details_value= [(self.conveyor.job_details.get(self.conveyor.conveyor[job_window], [])) for job_window in window_sections]
            
            # remaining operation bergeser sesuai window size
            for j, value in enumerate(job_details_value):
                next_observation_all[i, self.state_first_job_operation_location[j]] = value[0] if len(value)>0 else 0
                next_observation_all[i, self.state_second_job_operation_location[j]] = value[1] if len(value)>1 else 0


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

        reward_wait_all = self.reward_wait(actions, self.is_action_wait_succeed)
        reward_working_all = self.reward_working(self.observation_all, self.is_status_working_succeed )
        reward_step_all = self.reward_complete()
        reward_accept_all = self.reward_accept(self.observation_all)
        reward_agent_all=-0.75+reward_wait_all+reward_working_all+reward_step_all+reward_accept_all
        # print("reward_wait_all: ", reward_wait_all)
        # print("reward_working_all: ", reward_working_all)
        # print("reward_step_all: ", reward_step_all)
        # print("reward_agent_all: ", reward_agent_all)
        done_step = self.step_count >= self.max_steps
        truncated_step = True if len(self.conveyor.product_completed)>= self.n_jobs else False
        self.observation_all=next_observation_all
        info_step = {"actions": actions}

        return next_observation_all, reward_agent_all, done_step, truncated_step, info_step
    

    def reward_accept(self, observations, k_accept=0.5):
        rewards=[]
        for r, agent in enumerate(self.agents):
            obs=observations[r]
            if obs[self.state_status_location_all[r]]==1:
                rewards.append(k_accept)
            else:
                rewards.append(0)
        return rewards
    
    def reward_wait(self, actions,  is_action_wait_succeed,k_wait=0.5):
        rewards=[]
        for i, agent in enumerate(self.agents):
            
            if (actions[i]==1 or actions[i]==2) and is_action_wait_succeed[i]:
                if actions[i]==1:
                    factor_x=2.0
                elif actions[i]==2:
                    factor_x=3.0
                else:
                    print("FAILED ACTION: actions is not 1 or 2")
                rewards.append(float(agent.speed)/np.multiply(factor_x, sum(self.agent_speeds)))
            else:
                rewards.append(0)
        return np.multiply(k_wait,rewards)


    def reward_working(self, observations, is_status_working_succeed , k_working=2.5, k_energy=1):
        rewards=[]
        for r, agent in enumerate(self.agents):
           # obs=observations[r]
            if self.is_status_working_succeed[r]:
                rewards.append(float(agent.speed)/sum(self.agent_speeds)-k_energy)
            else:
                rewards.append(0)
        return np.multiply(k_working,rewards)
    
    def reward_complete(self, k_complete=10):
        value = k_complete*self.total_process_done
        self.total_process_done=0
        return value

    def render(self):
       # print(f"Time Step: {self.step_count}")
        # print("\nNEXT STATE RENDER:")
        for a, agent in enumerate(self.agents):
            #print("self.conveyor.job_details:, ", self.conveyor.job_details)
            print(f"Status Agent {agent.id} at position {int(self.observation_all[a][0])}: {int(self.observation_all[a][self.state_status_location_all[a]]) }")
            print("window product: ", agent.window_product, "\nworkbench: ", agent.workbench)
        #     if agent.processing_time_remaining>0:
        #         print("agent.processing_time_remaining: ", agent.processing_time_remaining)
        #     print("\n")
        self.conveyor.display()
        #print("-" * 50)

