import gymnasium as gym
import numpy as np

def rule_based_policy(state):
    """
    state = [x, y, vx, vy, angle, angularVel, legContact1, legContact2]
    
    Return nilai aksi (0, 1, 2, atau 3) berdasarkan aturan sederhana:
    - 0: diam
    - 1: engine kiri
    - 2: engine utama (ke atas)
    - 3: engine kanan
    """
    x, y, vx, vy, angle, angularVel, leg1, leg2 = state

    # 1. Tentukan sudut target (angle_targ) agar lander cenderung 'menghadap' titik (0,0).
    #    Misal kita memanfaatkan X (posisi horizontal) dan kecepatan horizontal untuk memberi sinyal seberapa
    #    besar sudut yg kita inginkan.
    angle_targ = 0.5 * x + 1.0 * vx
    #  Batasi agar tidak terlalu ekstrem, misal ±0.4 rad (~23 derajat)
    angle_targ = np.clip(angle_targ, -0.4, 0.4)

    # 2. Tentukan ketinggian target (hover_targ) agar semakin jauh dari pusat secara horizontal,
    #    kita sedikit "hover" lebih tinggi. Ini hanya salah satu strategi agar lander mengoreksi lebih cepat.
    hover_targ = 0.55 * abs(x)
    #  Hover ini kemudian dibandingkan dengan y dan vy.

    # 3. Hitung error sudut (angle) dan error ketinggian/kecepatan (hover).
    angle_error = angle_targ - angle
    angle_todo  = 0.5 * angle_error - 1.0 * angularVel

    hover_error = hover_targ - y
    hover_todo  = 0.5 * hover_error - 0.5 * vy

    # 4. Jika kaki sudah menyentuh tanah, lebih baik jangan terlalu agresif mengoreksi sudut lagi.
    #    Cukup kurangi kecepatan jatuh (vy).
    if leg1 == 1.0 or leg2 == 1.0:
        angle_todo = 0.0
        hover_todo = -0.5 * vy  # sekadar mengurangi benturan

    # 5. Terjemahkan "angle_todo" dan "hover_todo" menjadi aksi diskret:
    #    - Aksi 2 (engine utama) jika kita butuh dorongan ke atas lebih besar daripada dorongan untuk memutar.
    #    - Aksi 1 (engine kiri)  jika lander harus berotasi ke kanan (angle_todo > 0).
    #    - Aksi 3 (engine kanan) jika lander harus berotasi ke kiri  (angle_todo < 0).
    #    - Aksi 0 jika tidak ada tuntutan berarti (mungkin sudah cukup stabil).
    # 
    #    Kita bisa membuat aturan sederhana:
    #    - Jika hover_todo cukup besar (> 0.05) dan melebihi |angle_todo|,
    #      kita prioritaskan menyalakan engine utama (aksi 2).
    #    - Kalau angle_todo > 0.05 => aksi 1 (engine kiri).
    #    - Kalau angle_todo < -0.05 => aksi 3 (engine kanan).
    #    - Sisanya => aksi 0 (diam).

    main_threshold = 0.05
    angle_threshold = 0.05

    if (hover_todo > abs(angle_todo)) and (hover_todo > main_threshold):
        return 2  # engine utama
    elif angle_todo < -angle_threshold:
        return 3  # engine orientasi kanan
    elif angle_todo > angle_threshold:
        return 1  # engine orientasi kiri
    else:
        return 0   # diam (NOP)


env = gym.make("LunarLander-v3", render_mode='human')  # Env diskret
def run_rule_based_lander(num_episodes=5, render=True, seed=42):
    rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = rule_based_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

           
            env.render()

        rewards.append(episode_reward)
        print(f"Episode {episode+1}, total_reward={episode_reward:.2f}")

    env.close()
    return rewards


if __name__ == "__main__":
    # Jalankan contoh
    all_rewards = run_rule_based_lander(num_episodes=100, render=True)
    print("Rata-rata reward:", np.mean(all_rewards))
