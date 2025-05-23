import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_2 import CircularConveyor

# ============================================================================
# Agent: Merepresentasikan robot dengan workbench
# ============================================================================
class Agent:
    def __init__(self, agent_id: int, position: int, operation_capability : list, speed : float, base_processing_time :float, window_size: int):
        # Atribut state
        self.position = position  # Fixed position pada conveyor (indeks 0-based)
        self.operation_capability = np.array(operation_capability)  # Kemampuan operasi (misal [1,2])
        self.operation_now = 0   # Operasi berikutnya (jika ada)
        self.status = 0          # 0 = idle, 1 = processing
        self.remaining_operation = np.zeros(window_size)  # Window observasi
        # Properti tetap
        self.id = agent_id
        self.speed = speed
        self.base_processing_time = base_processing_time
        self.current_job = None
        self.processing_time_remaining = 0
        self.pending_reinsertion = False

    def build_state(self):
        """
        Membangun representasi state sebagai numpy array.
        Vektor state: [position, operation_capability(2), operation_now, status, window(3)]
        Total dimensi = 1 + 2 + 1 + 1 + 3 = 8.
        """
        return np.hstack([
            np.array([self.position]),
            self.operation_capability,
            np.array([self.operation_now, self.status]),
            self.remaining_operation
        ])

    def processing_time(self):
        return np.array(np.ceil(self.base_processing_time / self.speed))

    def start_job(self):
        self.processing_time_remaining = math.ceil(self.base_processing_time / self.speed)
        self.status = 1  # Mulai memproses

    def process(self, job_details):
        """
        Mengurangi waktu pemrosesan setiap timestep.
        Jika waktu pemrosesan habis, operasi selesai.
        """
        if self.current_job is not None:
            if self.processing_time_remaining > 0:
                self.processing_time_remaining -= 1
                return None
            if self.processing_time_remaining == 0:
                # Selesaikan operasi: keluarkan operasi pertama dari job.
                if self.current_job in job_details and len(job_details[self.current_job]) > 0:
                    return job_details[self.current_job].pop(0)
        return None

# ============================================================================
# FJSPEnv: Gymnasium Environment untuk Flexible Job Shop
# ============================================================================
class FJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, window_size: int, num_agents: int, max_steps: int):
        super(FJSPEnv, self).__init__()
        self.window_size = window_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0

        # Parameter Conveyor
        self.num_sections = 12
        self.max_capacity = 0.75
        self.arrival_rate = 0.3
        self.n_jobs = 20

        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, num_agents, n_jobs=self.n_jobs)

        # Konfigurasi agen
        # Fixed positions (indeks 0-based): Agent1:3, Agent2:7, Agent3:11
        self.agent_positions = [3, 7, 11]
        self.agent_operation_capability = [[1,2], [2,3], [1,3]]
        self.agent_speeds = [1, 2, 0.5]  # Agent2 2x lebih cepat; Agent3 2x lebih lambat
        self.base_processing_times = [6, 4, 2]  # Contoh waktu dasar untuk tiap agen
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                base_processing_time=self.base_processing_times[i],
                window_size=self.window_size
            )
            self.agents.append(agent)

        # Ruang observasi: tiap agen memiliki state vektor berukuran 8.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, 3+2+self.window_size), dtype=np.int16)

        # ---------------------------------------------------------------------
        # Perbaikan di sini: kita definisikan aksi sesuai gambar:
        # A0 = ACCEPT, A1..A(w-1) = WAIT (disederhanakan jadi 1 wait),
        # Ad = DECLINE, Ac = CONTINUE
        # Sehingga total 4 aksi: 0=ACCEPT, 1=WAIT, 2=DECLINE, 3=CONTINUE
        # ---------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)  # <-- PERUBAHAN

    def _get_obs(self):
        # Update state masing-masing agen dengan memanggil build_state()
        obs = []
        for agent in self.agents:
            obs.append(agent.build_state())
        return np.array(obs)

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
                base_processing_time=self.base_processing_times[i],
                window_size=self.window_size
            )
            self.agents.append(agent)
        return self._get_obs(), {}

    def step(self, actions):
        """
        actions: array dengan panjang num_agents, masing-masing aksi dalam {0,1,2,3}
          0: ACCEPT  ambil job di yr position
          1: WAIT    menunggu  untuk section conveyor sebelum yr dan sesuai panjang window size -1
          2: DECLINE menolak job di yr position
          3: CONTINUE default jika sedang memproses/tidak ada job
        """
        
        self.step_count += 1

        # 1. Generate job baru (jika kurang dari kapasitas maksimum) dan gerakkan conveyor.
        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()

        # 2. Untuk tiap agen yang sedang idle, periksa job di fixed position.
        for i, agent in enumerate(self.agents):
            print("Agent-", i, end=": ")
            yr = agent.position
            job_at_yr = self.conveyor.conveyor[yr]

            # Hanya dicek kalau agent idle dan ada job tepat di posisi agent
            if job_at_yr is not None and agent.current_job is None:
                if actions[i] == 0:  # ACCEPT
                    req_ops = self.conveyor.job_details.get(job_at_yr, [])
                    print("req_ops: ", req_ops)
                    if len(req_ops) > 0:
                        req_op = req_ops[0]
                        if req_op in agent.operation_capability:
                            agent.current_job = job_at_yr
                            self.conveyor.conveyor[yr] = None
                            agent.start_job()
                        # Jika tidak kompatibel, maka tidak diambil.
                elif actions[i] == 2:  # DECLINE <-- PERUBAHAN (dulunya ==0)
                    # Hapus job dari fixed position (dianggap menolak)
                    #self.conveyor.conveyor[yr] = None
                    pass
                # Jika aksi == 1 (WAIT) atau 3 (CONTINUE), tidak ada perubahan.
                elif actions[i] == 1:
                    pass
                elif actions[i] == 3:
                    pass
            else:
                print()

        # 3. Proses job yang ada di workbench masing-masing agen.
        for agent in self.agents:
            if agent.current_job is not None:
                op_completed = agent.process(self.conveyor.job_details)
                if op_completed is not None:
                    # Selesai satu operasi
                    # Jika job masih ada operasi tersisa, kembalikan job ke conveyor.
                    yr = agent.position
                    if self.conveyor.conveyor[yr] is None:
                        self.conveyor.conveyor[yr] = agent.current_job
                        # Jika job tidak ada operasi lagi, berarti selesai
                        if len(self.conveyor.job_details[agent.current_job]) == 0:
                            self.conveyor.product_completed.append(agent.current_job)
                            self.conveyor.conveyor[yr] = None
                        agent.current_job = None
                        agent.processing_time_remaining = 0
                    else:
                        # Jika slot penuh, agen menunggu
                        pass

        obs = self._get_obs()
        # Contoh reward: negatif dari panjang buffer.
        reward = -len(self.conveyor.buffer_jobs)
        done = self.step_count >= self.max_steps
        truncated = False
        info = {"actions": actions}

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
       # print(f"Time Step: {self.step_count}")
        self.conveyor.display()
        for agent in self.agents:
            status = agent.current_job if agent.current_job is not None else "Idle"
            print(f"Agent {agent.id} at position {agent.position}: {status}")
        #print("-" * 50)

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=50)
    obs, info = env.reset(seed=42)
    env.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial Observation:", obs)
    while not done and not truncated:
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        #actions = env.action_space.sample()
        actions=[0]*3
        
        print("Actions:", actions)
        obs, reward, done, truncated, info = env.step(actions)
        print("Reward:", reward)

        env.render()
        total_reward += reward
        print("-" * 50)
    print("Episode complete. Total Reward:", total_reward)
