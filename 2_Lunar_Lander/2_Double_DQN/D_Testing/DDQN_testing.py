import gymnasium as gym
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Environment settings
ENV_NAME = "LunarLander-v3"
RENDER_MODE = None
EPISODE_START = 400

# Initialize environment
env = gym.make(
    ENV_NAME,
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=0,
    turbulence_power=0,
    render_mode=RENDER_MODE
)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM =4
print(f"State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")

# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
#MODEL_DIR = os.path.join(PARENT_DIR, "B_HITL", "lunar_lander_hitl_2")
#MODEL_DIR = os.path.join(PARENT_DIR, "C_Human_Guided", "HG_saved_models_1", "HG_saved_models_1_part2")
MODEL_DIR = os.path.join(PARENT_DIR, "C_Human_Guided", "HG_saved_models_2")

class DDQNLoader:
    def __init__(self, state_dim, action_dim, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir

        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()

    def create_dqn_network(self):
        """Membuat model Q-network dengan layer fully connected."""
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def load_models(self, episode):
        """Memuat bobot model DQN dan target DQN dari file .h5."""
        dqn_load_path = os.path.join(self.load_dir, f'dqn_episode_{episode}.h5')
        target_dqn_load_path = os.path.join(self.load_dir, f'target_dqn_episode_{episode}.h5')

        for path in [dqn_load_path, target_dqn_load_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.dqn_network.load_weights(dqn_load_path)
        self.target_dqn_network.load_weights(target_dqn_load_path)
        print(f'Models loaded from episode {episode}')

        return self.dqn_network, self.target_dqn_network

# Load models
loader = DDQNLoader(state_dim=STATE_DIM, action_dim=ACTION_DIM, load_dir=MODEL_DIR)
dqn_network, target_dqn_network = loader.load_models(episode=EPISODE_START)
bias_output = dqn_network.layers[-1].get_weights()[-1]
print("bias_output:", bias_output)

# Action selection
def select_action(state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = dqn_network(state_tensor)
        return int(tf.argmax(q_values[0]).numpy())

# Running environment
def run_env(num_episodes, render):
    rewards = {}

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=1000+episode)
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            action = select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            #env.render()
        print("info:", info)

        rewards[episode] = episode_reward
        print(f"Episode {episode}, Total Reward = {episode_reward:.2f}")

    env.close()

    # Save rewards to JSON
    #file_path = os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HITL_without_noise.json")
    #file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HITL_noise.json")
    #file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HG_without_noise.json")
    #file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HG_noise.json")
    #file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HG_noise_2.json")
    file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_HG_without_noise_2.json")
    with open(file_path, "w") as f:
        json.dump(rewards, f, indent=4)

    return rewards

if __name__ == "__main__":
    all_rewards = run_env(num_episodes=100, render=False)
    print("Average Reward:", np.mean(list(all_rewards.values())))
