import gymnasium as gym
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Environment settings
ENV_NAME = "LunarLander-v3"
RENDER_MODE = "human"
N_EPISODES = 10
EPISODE_START = 400

# Initialize environment
env = gym.make(
    ENV_NAME,
    continuous=True,
    gravity=-10.0,
    enable_wind=False,
    wind_power=0,
    turbulence_power=0,
    render_mode=RENDER_MODE
)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
print(f"State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")

# Model directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "saved_models_2_final")

# Model creation functions
def create_actor_network():
    inputs = layers.Input(shape=(STATE_DIM,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(ACTION_DIM, activation='tanh')(x)  # Actions in range [-1, 1]
    return models.Model(inputs=inputs, outputs=outputs)

def create_critic_network():
    state_input = layers.Input(shape=(STATE_DIM,))
    action_input = layers.Input(shape=(ACTION_DIM,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=[state_input, action_input], outputs=output)

def model_creator():
    actor = create_actor_network()
    critic_1 = create_critic_network()
    critic_2 = create_critic_network()
    target_actor = create_actor_network()
    target_critic_1 = create_critic_network()
    target_critic_2 = create_critic_network()
    return actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2

# Load models
actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2 = model_creator()

def load_models(episode, dir_loc=MODEL_DIR):
    model_files = {
        "actor": f"actor_episode_{episode}.h5",
        "critic_1": f"critic_1_episode_{episode}.h5",
        "critic_2": f"critic_2_episode_{episode}.h5",
        "target_actor": f"target_actor_episode_{episode}.h5",
        "target_critic_1": f"target_critic_1_episode_{episode}.h5",
        "target_critic_2": f"target_critic_2_episode_{episode}.h5",
    }
    
    for model_name, file_name in model_files.items():
        path = os.path.join(dir_loc, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    actor.load_weights(os.path.join(dir_loc, model_files["actor"]))
    critic_1.load_weights(os.path.join(dir_loc, model_files["critic_1"]))
    critic_2.load_weights(os.path.join(dir_loc, model_files["critic_2"]))
    target_actor.load_weights(os.path.join(dir_loc, model_files["target_actor"]))
    target_critic_1.load_weights(os.path.join(dir_loc, model_files["target_critic_1"]))
    target_critic_2.load_weights(os.path.join(dir_loc, model_files["target_critic_2"]))

    print(f'Models loaded from episode {episode}')

load_models(EPISODE_START)
bias_output_actor = actor.layers[-1].get_weights()[-1]
print("Bias Output Actor:", bias_output_actor)

# Action selection
def select_action(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    action = actor(tf.expand_dims(state, axis=0), training=False)[0]
    return np.clip(action, -1, 1)

# Running environment
def run_env(num_episodes=10, render=True):
    rewards = {}

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=episode)
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            action = select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()

        rewards[episode] = episode_reward
        print(f"Episode {episode}, Total Reward = {episode_reward:.2f}")

    env.close()

    # Save rewards to JSON
    file_path = os.path.join(CURRENT_DIR, "TD3_testing_without_noise_cumulative_rewards.json")
    with open(file_path, "w") as f:
        json.dump(rewards, f, indent=4)

    return rewards

if __name__ == "__main__":
    all_rewards = run_env(num_episodes=100, render=True)
    print("Average Reward:", np.mean(list(all_rewards.values())))
