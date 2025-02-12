import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_2 import CircularConveyor
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
        # for i in range(self.num_agents):
        #     agent = Agent(
        #         agent_id=i+1,
        #         position=self.agent_positions[i],
        #         operation_capability=self.agent_operation_capability[i],
        #         speed=self.agent_speeds[i],
        #         window_size=self.window_size,
        #         num_agent=self.num_agents
        #     )
        #     self.agents.append(agent)

        # Ruang observasi: tiap agen memiliki state vektor berukuran 1+2+1+3+3 = 10
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
    

    def update_state(self, observation_all,  actions_all):
        next_observation_all=observation_all
        # next_observation_all=np.array(next_observation_all)

        
                
        for i, agent in enumerate(self.agents):
            print("\nAgent-", i+1, end=": ")
            #print("window product: ", agent.window_product, "\nworkbench: ", agent.workbench)
            observation=observation_all[i]
            status_location=self.state_status_location_all[i]
            #print("int(observation[self.state_yr_location])", int(observation[self.state_yr_location]))

            window_sections = [int(observation[self.state_yr_location]) - r for r in range(self.window_size)]
            print("window_sections: ", window_sections)
            agent.window_product=np.array(self.conveyor.conveyor)[window_sections]


            #capability_operation=[obs_first_operation, obs_second_operation]

            self.is_action_wait_succeed[i]=False
            self.is_status_working_succeed[i]=False
            

            if actions[i] == 0:
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
                        print("ACCEPT sudah di workbench")
                        self.is_job_moving_to_workbench[i]=False
                        agent.workbench=agent.buffer_job_to_workbench
                        agent.buffer_job_to_workbench={}

                        observation[status_location]=2 # accept menjadi working

                        list_operation=list(agent.workbench.values()) # karena [[1,2.3]] jadi [1,2,3]
                        print("list_operation: ", list_operation)

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

                            # processing time agent akan mulai dihitung
                            agent.processing_time_remaining = agent.processing_time(self.base_processing_times[select_operation-1])
                            print("agent.processing_time_remaining: ", agent.processing_time_remaining)
                        else:
                            print("FAILED ACTION: operation capability is False")
                
                    elif observation[status_location]==0  and observation[self.state_pick_job_window_location]== 1 and not self.is_job_moving_to_workbench[i] :
                        if  observation[self.state_first_job_operation_location[0]] in  observation[self.state_operation_capability_location]:
                            print("ACCEPT di conveyor yr")
                            # idle menjadi accept
                            observation[status_location]=1 
                            # job akan dipindahkan ke workbench
                            self.is_job_moving_to_workbench[i]=True
                            # menyimpan job conveyor yr di buffer untuk dipindahkan ke workbench pada timestep selanjutnya
                            agent.buffer_job_to_workbench["%s"%agent.window_product[0]]= observation[self.state_first_job_operation_location[0]]
                            # conveyor pada yr  akan kosong
                            self.is_job_conveyor_yr_remove=True
                            self.conveyor.conveyor[int(observation[self.state_yr_location])] = None
                            print("self.conveyor.conveyor: ", self.conveyor.conveyor)
                        else:
                            print("FAILED ACTION: observation first job operation is not in operation capability")
                    else:
                        print("FAILED ACTION: agent status is not idle or job is not in conveyor")

                else:
                    print("FAILED ACTION: workbench is not Empty.")


            elif actions[i] == 1:
                if observation[status_location]==0:
                    if observation[self.state_first_job_operation_location[1]] !=0:
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
                        observation[self.state_pick_job_window_location]=1
                        self.is_action_wait_succeed[i]=True
                        pass
                    else:
                        print("FAILED ACTION: there is no any job in conveyor yr-1")
                        pass
                else:
                    print("FAILED ACTION: agent status is not idle")

            elif actions[i] == 2:
                if observation[status_location]==0:
                    if observation[self.state_first_job_operation_location[2]] !=0:
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
                        observation[self.state_pick_job_window_location]=2
                        self.is_action_wait_succeed[i]=True
                        pass
                    else:
                        print("FAILED ACTION: there is no any job in conveyor yr-2")
                        pass
                else:
                    print("FAILED ACTION: agent status is not idle")

            elif actions[i] == 3:
                print("DECLINE")
                observation[self.state_pick_job_window_location]=0
                pass

            elif actions[i] == 4:
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
                    print("CONTINUE in working")
                    # agent sedang bekerja
                    self.is_status_working_succeed[i]=True
                    if agent.processing_time_remaining > 0:
                        agent.processing_time_remaining -= 1
                        print("agent.processing_time_remaining: ", agent.processing_time_remaining)
                    elif agent.processing_time_remaining == 0:
                        observation[status_location]= 3 # working menjadi completing
                        self.is_status_working_succeed[i]=False # operasi selesai dan agent tidak bekerja
                        print("processing_time_remaining is 0")
                        print("observation[status_location]: ", observation[status_location])

                    else:
                        print("FAILED ACTION: agent.processing_time_remaining")

                elif observation[status_location]==1: 
                    print("CONTINUE in idle")
                    pass
        
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
                    print("halo semua")
                    # jika conveyor pada yr kosong, maka product akan dikembalikan ke conveyor
                    if self.conveyor.conveyor[int(observation[self.state_yr_location])] is None:
                        # mengembalikan product ke conveyor
                        print("berhasil masuk sini")
                        self.conveyor.conveyor[int(observation[self.state_yr_location])]=str(list(agent.workbench)[0])
                        print("self.conveyor.conveyor: ", self.conveyor.conveyor)
                        self.product_return_to_conveyor[i]= False
                        # mengosongkan workbench -> agent menjadi idle -> tidak ada operasi pada robot
                        agent.workbench={}
                        observation[status_location]=0
                        observation[1+self.agent_many_operations]=0
                    else:
                        print("CAN'T RETURN: conveyor yr is not empty")
            
            if observation[status_location]==3 and not self.product_return_to_conveyor[i]:
                print("COMPLETING masuk sini")
                print("agent.workbench before: ", agent.workbench)
                self.total_process_done+=1 # menambahkan total process yang sudah selesai
                for key, vektor in agent.workbench.items():
                    # mengurangi operation yang sudah selesai
                    if len(vektor)>1:
                        agent.workbench[key].pop(0)
                    else:
                        # menghapus product yang sudah selesai dan mengosongkan workbench
                        self.conveyor.product_completed.append(key)
                        agent.workbench={} 
                        #print("self.conveyor.job_details: ", self.conveyor.job_details)
                        #print("agent.workbench after: ", agent.workbench)
                    # mengembalikan ke conveyor jika operasi product belum selesai
                    if agent.workbench:
                        self.product_return_to_conveyor[i]=True
                        
                    # mengupdate job_details
                    try:
                        self.conveyor.job_details[key]= agent.workbench[key]
                    except:
                        self.conveyor.job_details.pop(key)
                
                    #print("self.conveyor.job_details: ", self.conveyor.job_details)
                print("agent.workbench after: ", agent.workbench)

                # menyimpan job ke buffer untuk iterasi selanjutnya agar dapat dipindahkan ke conveyor
                #agent.buffer_job_to_conveyor=agent.workbench
                

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
          1: WAIT    menunggu  untuk section conveyor yr-1 hingga yr-window_size+1
          2: DECLINE menolak job di yr position dan tidak menunggu yr-1 hingga yr-window_size+1
          3: CONTINUE default jika sedang memproses/tidak ada job
        """
        
        self.step_count += 1


        next_observation_all= self.update_state(observation_all=self.observation_all, actions_all=actions)
        print()

        reward_wait_all = self.reward_wait(actions, self.is_action_wait_succeed)
        reward_working_all = self.reward_working(self.observation_all, self.is_status_working_succeed ,factor_x=1)
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
            if obs[self.state_status_location_all[r]]==2 and is_status_working_succeed[r]:
                rewards.append(agent.speed/sum(factor_x*self.agent_speeds))
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


def FAA_action(states):
    actions=[]
    for i, state in enumerate(states):
        print("aa: ", state[env.state_first_job_operation_location[2]])

        if state[env.state_status_location_all[i]]==1 and state[env.state_operation_now_location]==0: # agent accept dan operasi pekerjaan belum di assign
            actions.append(0)
        elif state[env.state_status_location_all[i]]==2 and state[env.state_operation_now_location]!=0: # agent working
            actions.append(4)

        elif state[env.state_first_job_operation_location[0]]!=0: # ada job di yr
            actions.append(0)
        elif state[env.state_first_job_operation_location[1]]!=0:# ada job di yr-1
            actions.append(1) 
        elif state[env.state_first_job_operation_location[2]]!=0: # ada job di yr-2
            actions.append(2)
        else:
            actions.append(4) # continues
    return actions

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=25)
    state, info = env.reset(seed=42)
    #nv.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial state:", state)
    while not done and not truncated:
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        #actions = env.action_space.sample()

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
    print("Episode complete. Total Reward:", total_reward)
