import gymnasium as gym
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from env_5c_4 import FJSPEnv  
from MASKING_ACTION_MODEL import masking_action
# Environment settings
ENV_NAME = "LunarLander-v3"
RENDER_MODE = None
EPISODE_START = 500

STATE_DIM = 14
ACTION_DIM = 5
print(f"State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")

# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(PARENT_DIR,'main_dir', "saved_3")


class DDQN_model:
    def __init__(self, state_dim, action_dim, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir

        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()

    def create_dqn_network(self):
        # Dense network: input shape is (state_dim,)
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
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
loader = DDQN_model(state_dim=STATE_DIM, action_dim=ACTION_DIM, load_dir=MODEL_DIR)
dqn_network, target_dqn_network = loader.load_models(episode=EPISODE_START)
bias_output = dqn_network.layers[-1].get_weights()[-1]
print("bias_output:", bias_output)

def select_action_with_masking(state, action_mask):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)  # shape: (1, state_dim)
        q_values = dqn_network(state_tensor)  # shape: (1, action_dim)
        action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
        masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
        return int(tf.argmax(masked_q_values, axis=1)[0])

# Running environment
def run_env(num_episodes, render):
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000)
    num_episodes = 10
    rewards = {}
    for episode in range(1, num_episodes + 1):
        state, info = env.reset(seed=1000+episode)
        episode_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            mask_actions = masking_action(state, env)
            joint_actions = []
            for single_state, mask in zip(state, mask_actions):
                action = select_action_with_masking(single_state, mask)
                joint_actions.append(action)
            
            joint_actions = np.array(joint_actions)
            next_state, reward, done, truncated, info =  env.step(joint_actions)
            episode_reward +=  np.mean(reward)
            state = next_state
            
            if env.FAILED_ACTION:
                print("FAILED ENV")
                break
        
        if env.FAILED_ACTION:
            print("FAILED ENV")
            break



        rewards[episode] = episode_reward
        print(f"Episode {episode}, Total Reward = {episode_reward:.2f}", "jumlah step:", env.step_count)

    env.close()

    # Save rewards to JSON

    file_path= os.path.join(CURRENT_DIR, "Testing_cumulative_rewards_DDQN_1.json")
    with open(file_path, "w") as f:
        json.dump(rewards, f, indent=4)

    return rewards

if __name__ == "__main__":
    all_rewards = run_env(num_episodes=100, render=False)
    print("Average Reward:", np.mean(list(all_rewards.values())))
