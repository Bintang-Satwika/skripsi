import gymnasium as gym
import numpy as np
import time

# --- Implementasi Controller PID untuk Lunar Lander ---
class LunarLanderPIDPolicy:
    def __init__(self, dt=0.1):
        """
        Inisialisasi PID controller untuk kontrol sudut (angle) dan ketinggian (hover).

        Parameter:
        - dt: waktu sampling (delta t) antara tiap update.
        """
        # Koefisien PID untuk kontrol sudut
        self.kp_angle = 0.5   # Proporsional
        self.ki_angle = 0.0   # Integral (bisa diubah jika diperlukan)
        self.kd_angle = 1.0   # Derivatif

        # Koefisien PID untuk kontrol ketinggian (hover)
        self.kp_hover = 0.5   # Proporsional
        self.ki_hover = 0.0   # Integral
        self.kd_hover = 0.5   # Derivatif

        self.dt = dt

        # Inisialisasi nilai integral dan error sebelumnya untuk masing-masing PID
        self.integral_angle = 0.0
        self.prev_error_angle = 0.0

        self.integral_hover = 0.0
        self.prev_error_hover = 0.0

    def reset(self):
        """Reset nilai integral dan error sebelumnya, misalnya saat memulai episode baru."""
        self.integral_angle = 0.0
        self.prev_error_angle = 0.0
        self.integral_hover = 0.0
        self.prev_error_hover = 0.0

    def get_action(self, state):
        """
        Menghitung aksi diskret (0, 1, 2, atau 3) berdasarkan PID control.
        
        Parameter:
        - state: list dengan format [x, y, vx, vy, angle, angularVel, legContact1, legContact2]
        
        Return:
        - Aksi diskret:
          0: tidak ada aksi (NOP)
          1: engine kiri (rotasi ke kanan)
          2: engine utama (dorongan ke atas)
          3: engine kanan (rotasi ke kiri)
        """
        # Ekstrak state
        x, y, vx, vy, angle, angularVel, leg1, leg2 = state

        # --- 1. Tentukan target sudut dan ketinggian ---
        angle_targ = 0.5 * x + 1.0 * vx
        angle_targ = np.clip(angle_targ, -0.4, 0.4)  # batasi hingga ±0.4 radian (~23°)

        hover_targ = 0.55 * abs(x)

        # --- 2. Hitung error masing-masing ---
        error_angle = angle_targ - angle
        error_hover = hover_targ - y

        # Update komponen integral
        self.integral_angle += error_angle * self.dt
        self.integral_hover += error_hover * self.dt

        # Hitung komponen derivatif
        derivative_angle = (error_angle - self.prev_error_angle) / self.dt
        derivative_hover = (error_hover - self.prev_error_hover) / self.dt

        # --- 3. Hitung output PID ---
        output_angle = (self.kp_angle * error_angle +
                        self.ki_angle * self.integral_angle +
                        self.kd_angle * derivative_angle)

        output_hover = (self.kp_hover * error_hover +
                        self.ki_hover * self.integral_hover +
                        self.kd_hover * derivative_hover)

        # Simpan error untuk langkah berikutnya
        self.prev_error_angle = error_angle
        self.prev_error_hover = error_hover

        # --- 4. Penyesuaian saat kaki menyentuh tanah ---
        if leg1 == 1.0 or leg2 == 1.0:
            output_angle = 0.0
            output_hover = -0.5 * vy

        # --- 5. Konversi output PID menjadi aksi diskret ---
        main_threshold = 0.05
        angle_threshold = 0.05

        if (output_hover > abs(output_angle)) and (output_hover > main_threshold):
            return 2  # engine utama
        elif output_angle < -angle_threshold:
            return 3  # engine kanan
        elif output_angle > angle_threshold:
            return 1  # engine kiri
        else:
            return 0  # tidak ada aksi (NOP)


# --- Uji Controller PID dalam Environment Lunar Lander ---
def main():
    # Buat environment Lunar Lander (pastikan menggunakan versi discrete, misalnya "LunarLander-v2")
    env = gym.make("LunarLander-v3", render_mode="human")
    policy = LunarLanderPIDPolicy(dt=0.0001)

    n_episodes = 5
    for episode in range(n_episodes):
        # Reset environment dan controller PID di awal episode baru
        state, info = env.reset(seed=episode)
        #policy.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Dapatkan aksi dari controller PID
            action = policy.get_action(state)
            # Lakukan step pada environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Delay sedikit untuk visualisasi (opsional)
            time.sleep(0.01)
            if done or truncated:
                break

        print(f"Episode {episode + 1} selesai dengan total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
