import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FJSPEnv(gym.Env):
    """
    Custom environment untuk Flexible Job Shop Scheduling Problem (FJSP).
    """

    def __init__(self, window_size=3, num_agents=3):
        super(FJSPEnv, self).__init__()
        # r adalah index dari agent
        # i adalah index dari product
        # j adalah index dari job
        # k adalah index dari processing step
        # l adalah index dari tipe operasi
        # q adalah index dari conveyor section

        # m adalah jumlah robot
        # n adalah jumlah pekerjaan yang harus diselesaikan sebelum termination
        # lambda adalah arrival rate dari job
        # p adalah jumlah product
        # o adalah jumlah unique operation types
        # c adalah kapasitas conveyor belt section
        # c_max adalah kapasitas conveyor belt dalam presentase


        # t adalah index dari waktu
        # y adalah index dari posisi agent
        # v adalah kecepatan agent
        # --------------------------------------------------
        # PARAMETER & KONFIGURASI LINGKUNGAN
        # --------------------------------------------------
        self.window_size = window_size
        self.num_agents = num_agents

        # parameter statis konstan
        self.c_conveyor_jumlah = window_size*num_agents
        self.m_robot_jumlah = num_agents
        self.n_job_jumlah = 20
        self.c_max = 0.75

        # --------------------------------------------------
        # ACTION SPACE
        # --------------------------------------------------
        # 0=decline, 1=accepted, 2=wait, 3=continue
        # Discrete(4) sebagai action space
        self.action_space = spaces.Discrete(4)

        # --------------------------------------------------
        # OBSERVATION SPACE
        # --------------------------------------------------
        # Agar sederhana, kita “flatten” state jadi satu vektor.
        # Misalnya, kita masukkan:
        #   - y_r: posisi agent (dimension = num_agents)
        #   - O_r_t: operasi aktif untuk tiap agent (dimension = num_agents)
        #   - Z_t: status agent [0=idle,1=accepted,2=working,3=done], di sini juga dimension = num_agents
        #   - S_hat_y_r_t: sisa operasi (tiap agent array length=window_size) => total num_agents * window_size
        #
        # Total length = num_agents + num_agents + num_agents + (num_agents*window_size)
        #              = 3*num_agents + (num_agents*window_size)
        obs_dim = 3*self.num_agents + self.num_agents*self.window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Variabel internal
        self.state = None
        self.step_count = 0
        self.max_steps = 50  # contoh batas langkah per episode

        # Inisialisasi environment
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Mengatur ulang environment ke kondisi awal.
        Menghasilkan state awal dan info dictionary (opsional).
        """
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._build_state()
        return self.state, {}

    def step(self, action):
        """
        Mengeksekusi aksi (action) dan mengembalikan:
          next_state, reward, done, truncated, info
        """
        self.step_count += 1

        # --------------------------------------------------
        # 1. Hitung reward
        # --------------------------------------------------
        # Anda bisa memasukkan formula reward yang lebih kompleks.
        # Contoh sederhana:
        #    - penalti -1 jika action=0 (decline)
        #    - +2 jika action=1 (accepted)
        #    - -0.2 jika action=2 (wait)
        #    - +1 jika action=3 (continue)
        if action == 0:
            reward = -1.0
        elif action == 1:
            reward = 2.0
        elif action == 2:
            reward = -0.2
        else:  # action == 3
            reward = 1.0

        # --------------------------------------------------
        # 2. Update state
        # --------------------------------------------------
        # Di sini kita buat contoh “random” agar tampak ada perubahan.
        # Anda perlu menyesuaikan logika agar merefleksikan realita FJSP.
        self.state = self._build_state()

        # --------------------------------------------------
        # 3. Tentukan apakah episode selesai (done)
        # --------------------------------------------------
        # Contoh: episode selesai jika step_count >= max_steps
        done = (self.step_count >= self.max_steps)
        truncated = False  # Atau atur sesuai kebutuhan

        info = {}
        return self.state, reward, done, truncated, info

    def _build_state(self):
        """
        - y_r: posisi agent
        - O_r_t: operasi aktif
        - Z_t: status
        - S_hat_y_r_t: sisa operasi (tiap agent array length=window_size)
        """
        # Posisi a la (5,5,5) => kita perumum:
        y_r = [self.window_size * (i + 1) for i in range(self.num_agents)]
    

        # O_r_t => misalnya acak {1,2,3}
        O_r_t = np.random.randint(1,4, size=self.num_agents)

        # Status => misal {0=idle,1=accepted,2=working,3=done}
        Z_t = np.random.randint(0,4, size=self.num_agents)

        # Sisa operasi => random choice misal [1,2,3], masing2 length=window_size
        S_hat_y_r_t = []
        for _ in range(self.num_agents):
            arr_ops = np.random.choice([1,2,3], size=self.window_size)
            S_hat_y_r_t.append(arr_ops)
        # Flatten sisa operasi
        S_hat_y_r_t = np.concatenate(S_hat_y_r_t)

        # Gabungkan jadi array 1D
        y_r = np.array(y_r, dtype=np.float32)
        O_r_t = np.array(O_r_t, dtype=np.float32)
        Z_t = np.array(Z_t, dtype=np.float32)
        S_hat_y_r_t = np.array(S_hat_y_r_t, dtype=np.float32)

        # state jadi  [y_r, O_r_t, Z_t, S_hat_y_r_t]
        return np.concatenate([y_r, O_r_t, Z_t, S_hat_y_r_t], axis=0)

# --------------------------------------------------
# Contoh penggunaan environment
# --------------------------------------------------
if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3)
    for _ in range(10):
        obs, info = env.reset(seed=1)
        print("Initial observation:", obs)
        done = False
        total_reward = 0
        while not done:
            # Contoh: random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            done = done or truncated
        
        print("Episode done. Total Reward:", total_reward)