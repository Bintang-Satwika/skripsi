import gymnasium as gym
import numpy as np
import os
import json

array_action = [0, 0]

def rule_based_policy(state):
    global array_action
    x, y, vx, vy, angle, angularVel, leg1, leg2 = state

    angle_targ = 0.5 * x + 1.0 * vx
    angle_targ = np.clip(angle_targ, -0.4, 0.4)

    hover_targ = 0.55 * abs(x)
    
    angle_error = angle_targ - angle
    angle_todo  = 0.5 * angle_error - 1.0 * angularVel

    hover_error = hover_targ - y
    hover_todo  = 0.5 * hover_error - 0.5 * vy

    if leg1 == 1.0 or leg2 == 1.0:
        angle_todo = 0.0
        hover_todo = -0.5 * vy

    main_threshold = 0.05
    angle_threshold = 0.05
    
    if (hover_todo > abs(angle_todo)) and (hover_todo > main_threshold):
        array_action[0] = 1
    else:
        array_action[0] = 0
        
    if angle_todo < -angle_threshold:
        array_action[1] = 1
    elif angle_todo > angle_threshold:
        array_action[1] = -1
    else:
        array_action[1] = 0

def run_rule_based_lander(num_episodes=10, render=True, seed=42, env_name="LunarLander-v3", render_mode=None):
    env = gym.make(
        env_name, 
        continuous=True,
        gravity=-10.0,
        enable_wind=False,
        wind_power=0,
        turbulence_power=0,
        render_mode=render_mode
    )
    rewards = {}

    for episode in range(1, num_episodes+1):
        obs, _ = env.reset(seed=1000+episode)
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            rule_based_policy(obs)
            obs, reward, done, truncated, info = env.step(array_action)
            episode_reward += reward

        rewards[episode] = episode_reward
        print(f"Episode {episode}, total_reward={episode_reward:.2f}")

    env.close()
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, "Testing_cumulative_rewards_ruled_based_without_noise_400episode.json")
    #file_path= os.path.join(current_directory, "Testing_cumulative_rewards_ruled_based_noise.json")
    
    with open(file_path, "w") as f:
        json.dump(rewards, f, indent=4)
    
    return rewards

if __name__ == "__main__":
    all_rewards = run_rule_based_lander(num_episodes=400, render=True)
    print("Rata-rata reward:", np.mean(list(all_rewards.values())))
