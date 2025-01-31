import gymnasium as gym
import numpy as np

def rule_based_policy_continuous_improved(state):
    """
    state = [x, y, vx, vy, angle, angularVel, legContact1, legContact2]

    Kita menghasilkan aksi berformat [main_engine, lateral_engine], masing-masing di [-1..+1].
    - main_engine < 0   => mesin utama OFF
    - main_engine in [0..1] => 50%-100% power (semakin besar, semakin mendekati 100%)
    - lateral_engine in (-0.5..0.5) => OFF
    - lateral_engine < -0.5 => booster kiri (50%-100%)
    - lateral_engine > 0.5  => booster kanan (50%-100%)
    """

    # Pisahkan state:
    x, y, vx, vy, angle, angularVel, leg1, leg2 = state

    # 1. Tentukan target sudut (angle_targ) dan ketinggian (hover_targ)
    #    agar lander cenderung ‘menghadap’ pusat (0,0).
    angle_targ = 0.4 * x + 1.0 * vx
    angle_targ = np.clip(angle_targ, -0.4, 0.4)  # batasi sudut target agar tidak ekstrem

    hover_targ = 0.55 * abs(x)  # makin jauh dari pusat, makin besar hover

    # 2. Hitung "error" sudut dan "error" hover.
    angle_error = angle_targ - angle
    angle_todo  = 0.5 * angle_error - 0.8 * angularVel

    hover_error = hover_targ - y
    hover_todo  = 0.5 * hover_error - 0.5 * vy

    # 3. Jika kaki sudah menyentuh tanah, kurangi agresivitas.
    if leg1 == 1.0 or leg2 == 1.0:
        angle_todo = 0.0
        # Sedikit dorongan ke atas jika masih jatuh kencang
        hover_todo = -0.3 * vy

    # 4. Konversi hover_todo => main_engine
    #    - Jika hover_todo > 0, kita menyalakan mesin dengan skala [0..1]
    #    - Jika hover_todo <= 0, matikan mesin (main < 0)
    main_engine = -1.0  # default: OFF
    if hover_todo > 0.05:  
        # kita skala agar  hover_todo ~ 1 => main ~ 1, 
        #                 hover_todo kecil => main mendekati 0 => (50% power).
        # Batasin max = 1
        power = 0.5 * np.clip(hover_todo, 0, 1.0)  
        # power di [0..0.5] => berarti real power [50%..100%].
        main_engine = np.clip(power, 0.0, 1.0)  

    # 5. Konversi angle_todo => lateral_engine
    #    - Jika angle_todo > 0 => kita perlu booster kanan (action > 0.5)
    #    - Jika angle_todo < 0 => booster kiri (action < -0.5)
    #    - Kecil => OFF
    side_power = 0.0
    side_threshold = 0.02
    if angle_todo > side_threshold:
        # Skala angle_todo agar jika angle_todo besar => side_power ~ 1.0
        val = 0.5 + 0.5*np.clip(angle_todo, 0, 1.0)  # [0.5..1]
        side_power = np.clip(val, 0.5, 1.0)
    elif angle_todo < -side_threshold:
        val = -0.5 + 0.5*np.clip(angle_todo, -1.0, 0) # [-1..-0.5]
        side_power = np.clip(val, -1.0, -0.5)
    else:
        side_power = 0.0  # OFF

    return np.array([main_engine, side_power], dtype=np.float32)


def run_rule_based_lander_continuous(num_episodes=5, render=True, seed=42, 
                                     env_name="LunarLander-v3", render_mode='human'):
    """
    Menjalankan environment continuous,
    menampilkan episode demi episode, dan mencetak total reward.
    """
    # Buat environment Continuous (continuous=True) dengan parameter lain:
    env = gym.make(
        env_name, 
        continuous=True,
        gravity=-10.0,
        enable_wind=True,       # boleh juga dimatikan jika ingin stabil
        wind_power=15.0,        # ini cukup besar => pendaratan lebih menantang
        turbulence_power=1.5,
        render_mode=render_mode
    )

    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = rule_based_policy_continuous_improved(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)
        print(f"Episode {episode+1}, total_reward={episode_reward:.2f}")

    env.close()
    return rewards


if __name__ == "__main__":
    # Eksekusi 5 episode
    all_rewards = run_rule_based_lander_continuous(num_episodes=5, render=True)
    print("Rata-rata reward:", np.mean(all_rewards))
