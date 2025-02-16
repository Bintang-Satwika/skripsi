import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FJSPEnv(gym.Env):
    """
    Contoh sederhana environment FJSP berbasis Gymnasium.
    Memastikan reproducibility dengan memakai self.np_random.
    """

    def __init__(self, window_size=3, num_agents=3, max_steps=10):
        super().__init__()
        # Parameter environment
        self.window_size = window_size
        self.num_agents = num_agents
        self.max_steps = max_steps

        # Contoh parameter statis
        self.c_conveyor_jumlah =  window_size*num_agents
        self.m_robot_jumlah = num_agents
        self.n_job_jumlah = 20
        self.c_max = 0.75
        self.p_product_jumlah = 4
        self.o_operation_jumlah = 3
        
        # kecepatan konstan agent r untuk setiap tipe operasi yang ada
        # baris = agent r, kolom = tipe operasi l
        self.v_r_l= np.array([[1,2,3], # agent 1
                              [1,3,5], # agent 2
                              [2,3,4]]) # agent 3
      
        
        print("v_r_l: ",self.v_r_l)
        # Action space: 0=decline,1=accepted,2=wait,3=continue
        self.action_space = spaces.Discrete(4)

        # Observation space (flatten): 
        # y_r (num_agents) + O_r_t (num_agents) + Z_t (num_agents) + S_hat_y_r_t (num_agents*window_size)
        obs_dim =  5 + self.window_size 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, obs_dim), dtype=np.float32
        )

        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """
        Mengatur ulang environment ke kondisi awal.
        Menghasilkan state awal dan info dictionary (opsional).
        """
        # Pemanggilan super().reset(seed=seed) akan mengatur self.np_random
        super().reset(seed=seed)

        # Set ulang step_count
        self.step_count = 0

        # Bangun state pertama
        self.state = self._build_state()

        # Kembalikan (observation, info)
        return self.state, {}

    def step(self, action):
        """
        Mengeksekusi aksi (action) dan mengembalikan:
          next_state, reward, done, truncated, info
        """
        self.step_count += 1

        # Contoh sederhana perhitungan reward
        # if action == 0:
        #     reward = -1.0      # decline
        # elif action == 1:
        #     reward = 2.0       # accepted
        # elif action == 2:
        #     reward = -0.2      # wait
        # else:  # action == 3
        #     reward = 1.0       # continue
        reward = self.reward_formula(self.state)

        # Perbarui state secara acak NANTI BUKAN INI TAPI DARI  ACTION
        self.state = self._build_state()

        # Tentukan apakah episode selesai
        done = (self.step_count >= self.max_steps)
        truncated = False  # atau sesuai logika yang diinginkan

        info = {}
        return self.state, reward, done, truncated, info

    def _build_state(self):
        """
        Bangun state (flatten) menggunakan self.np_random sebagai
        sumber acak agar reproducible bila seed ditentukan di reset().
        """
        # y_r: posisi seluruh agent berjarak window_size
        y_r = [self.window_size*(i+1) for i in range(self.num_agents)]
        O_r_a =[1,2,1] # O_r_a = operasi pertama yang bisa dilakukan oleh setiap agent 
        O_r_b =[2,3,3] # O_r_b = operasi kedua yang bisa dilakukan oleh setiap agent

        # O_r_t = operasi yang sedang dikerjakan oleh seluruh agent (acak 1..3)
        # O_r_t = self.np_random.integers(0, 4, size=self.num_agents)

        # tidak ada operasi yang dikerjakan saat reset environment
        O_r_t = np.zeros(self.num_agents, dtype=np.float32)
        O_r_t[0] = 1 # agent pertama sedang mengerjakan operasi pertama
        O_r_t[1] = 2 # agent kedua sedang mengerjakan operasi kedua
        O_r_t[2] = 3 # agent ketiga sedang mengerjakan operasi ketiga

        # Z_t =  status seluruh agent (acak 0..3)
        # Z_t = self.np_random.integers(0, 4, size=self.num_agents)

        # status agent idle saat reset environment
        Z_t = np.zeros(self.num_agents, dtype=np.float32)
        Z_t[0] = 2 # agent pertama sedang bekerja
        Z_t[1] = 3 # agent kedua sedang menunggu
        Z_t[2] = 2 # agent ketiga sedang bekerja

        # jumlah operasi yang belum dikerjakan untuk masing-masing product , tiap agent array length=window_size ∈ {1,2,3}
        S_hat_y_r_t = []
        for _ in range(self.num_agents):
            S_hat_y_r_t.append(self.np_random.integers(0, 2, size=self.window_size))
        S_hat_y_r_t = np.array(S_hat_y_r_t , dtype=np.float32)

        # Gabungkan semua state
        state = np.zeros((self.num_agents, 5 + self.window_size), dtype=np.float32)
        state[:, 0] = y_r # posisi seluruh agent
        state[:, 1] = O_r_a # operasi pertama yang bisa dilakukan oleh setiap agent
        state[:, 2] = O_r_b # operasi kedua yang bisa dilakukan oleh setiap agent
        state[:, 3] = O_r_t # operasi yang sedang dikerjakan oleh setiap agent
        state[:, 4] = Z_t # status seluruh agent
        state[:, 5:] = S_hat_y_r_t # sisa operasi yang harus dikerjakan oleh seluruh agent

        return state
    
    def reward_formula(self, state,  alpha=1.1, zeta=1, beta=1, gamma=0.5):
        R_step = 0
        R_process_all_agent=np.zeros(self.num_agents)
        R_wait_all = np.zeros(self.num_agents)
       # action_avaliable= [1] + [2] * (self.window_size - 1) + [0]+ [3]
       # print("action_avaliable: ", action_avaliable)
     
        # Sum of v_r_l values for each operation
        sum_v_each_operation = np.sum(self.v_r_l, axis=0)
        print("sum_v_each_operation: ", sum_v_each_operation)

        # Calculate R_process for all agents
        mask_process= state[:, 4] == 2  # Mask for agents with status 2 (working)
        mask_process_index_true =  np.where(mask_process)[0]
        print("mask: ",mask_process)
        print("mask_true: ",mask_process_index_true)
        print("state[:, 3]: ",state[:, 3])
        print('self.v_r_l[mask_process_index_true ]:', self.v_r_l[mask_process_index_true])
      
        a=np.where(state[mask_process_index_true , 3] == 1, self.v_r_l[mask_process_index_true, 0] /sum_v_each_operation[0], 0)
        print("a: ",a)
        b=np.where(state[mask_process_index_true , 3] == 2, self.v_r_l[mask_process_index_true, 1] /sum_v_each_operation[1], 0)
        print("b: ",b)
        c=np.where(state[mask_process_index_true , 3] == 3, self.v_r_l[mask_process_index_true, 2] /sum_v_each_operation[2], 0)
        print("c: ",c)
        R_process_all_agent[mask_process_index_true]= a+b+c
        print("R_process_all_agent: ",R_process_all_agent, end="\n\n")
     



        mask_wait= state[:, 4] == 3  # Mask for agents with status 3 (waiting)
        mask_wait_index_true =  np.where(mask_wait)[0]
        print("mask_wait: ",mask_wait)
        print("mask_true: ",mask_wait_index_true)
        print("state[:, 3]: ",state[:, 3])
        print('self.v_r_l[mask_wait_index_true]:', self.v_r_l[mask_wait_index_true])

        x_distance=np.array([1,1,1])
        print("aaa: ",self.v_r_l[mask_wait_index_true, 2])

        a=np.where(state[mask_wait_index_true, 3] == 1, 
                   x_distance[mask_wait_index_true]*self.v_r_l[mask_wait_index_true, 0] /sum_v_each_operation[0], 0)
        print("a: ",a)
        b=np.where(state[mask_wait_index_true, 3] == 2,  
                   x_distance[mask_wait_index_true]*self.v_r_l[mask_wait_index_true, 1] /sum_v_each_operation[1], 0)
        print("b: ",b)
        c=np.where(state[mask_wait_index_true, 3] == 3,  
                   x_distance[mask_wait_index_true]*self.v_r_l[mask_wait_index_true, 2] /sum_v_each_operation[2], 0)
        print("c: ",c)
        R_wait_all[mask_wait_index_true]= a+b+c
        print("R_wait_all: ",R_wait_all)

    
        total_reward = -alpha + zeta*R_step + beta*R_process_all_agent+ gamma*R_wait_all
        return total_reward

    def action_env(self, state):
        action_avaliable= [1] + [2] * (self.window_size - 1) + [0]+ [3]
        print("action_avaliable: ", action_avaliable)
        return action_avaliable

    def emmiter_product(self):
        np.random.poisson(lam=0.0030, size=1)
        pass

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3)
    for _ in range(1):
        obs, info = env.reset(seed=None)
        print("Initial observation:", obs)
        done = False
        total_reward = 0
        print(env.observation_space.shape)
        for _ in range(1):
            print("\n")
            # Contoh: random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            done = done or truncated
        
        print("Episode done. Total Reward:", total_reward)