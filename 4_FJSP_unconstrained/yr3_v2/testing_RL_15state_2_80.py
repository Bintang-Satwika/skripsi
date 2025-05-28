import gymnasium as gym
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from env_1_testing_80 import FJSPEnv  
from RULED_BASED import MASKING_action
# Environment settings
RENDER_MODE = None
EPISODE_START =  600

STATE_DIM = 15
ACTION_DIM = 6
print(f"State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")

# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
print("PARENT_DIR:", PARENT_DIR)
MODEL_DIR = os.path.join(PARENT_DIR,'yr3_v2', "DQN_yr3_5")


class DDQN_model:
    def __init__(self, state_dim, action_dim, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir
        self.num_agents = 3
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()

    # def create_dqn_network(self):

    #     state_input = layers.Input(shape=(self.num_agents, self.state_dim))
        
    #     # Apply Dense layers without activation and then apply LeakyReLU using TimeDistributed
    #     x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(state_input)
    #     x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
    #     output = layers.TimeDistributed(layers.Dense(self.action_dim, activation='linear'))(x)
        
    #     model = models.Model(inputs=state_input, outputs=output)
    #     return model
    def create_dqn_network(self):
        # For 3 agents: input shape becomes (num_agents, state_dim)
        state_input = layers.Input(shape=(self.num_agents, self.state_dim))
        
        
        # Apply Dense layers without activation and then apply LeakyReLU using TimeDistributed
        x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(state_input)
        x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
        output = layers.TimeDistributed(layers.Dense(self.action_dim, activation='linear'))(x)
        
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

@tf.function
def select_action_with_masking(state, action_mask_all):

    # Add batch dimension: state becomes (1, num_agents, state_dim)
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    q_values = dqn_network(state_tensor)  # shape (1, num_agents, action_dim)
    q_values = q_values[0]  # shape (num_agents, action_dim)
    action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)
    masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))

    actions = tf.argmax(masked_q_values, axis=1)

    return actions

def normalize(state):
    state_mins = np.array([3,  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    # Normalize each feature (column) using broadcasting.
    normalized_state = (state - state_mins) / (state_maxs - state_mins)
    return np.float32(normalized_state)

def run_env(num_episodes, render):
    env = FJSPEnv(window_size=4, num_agents=3, max_steps=1000, episode=1)
    num_episodes = 1000
    rewards = {}
    makespan = {}
    energy= {}
    success = {}
    success_energy = {}
    for episode in range(1, num_episodes+1):
        state, info = env.reset(seed=1000+episode)
        state= state[:, :-1]
        if (episode-1) %1 == 0:
            episode_seed= episode-1
        env.conveyor.episode_seed= episode_seed
        episode_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            mask_actions = MASKING_action(state, env)
            state_normalized = normalize(state)
            joint_actions = select_action_with_masking(state_normalized, mask_actions)
                
            joint_actions = np.array(joint_actions)

            next_state, reward, done, truncated, info =  env.step(joint_actions)
            next_state = next_state[:, :-1]
            episode_reward +=  np.mean(reward)
            state = next_state
            
            if env.FAILED_ACTION:
                print("FAILED ENV")
                break
        
        if env.FAILED_ACTION:
            print("FAILED ENV")
            break


        if len(env.conveyor.product_completed) <21 or  env.step_count >= 250:
            success[episode] = 0
        else:
            success[episode] = 1
        if len(env.conveyor.product_completed) <21 or  env.step_count >= 250 or sum(env.agents_energy_consumption)>=250:
            success_energy[episode] = 0
        else:
            success_energy[episode] = 1
        
        rewards[episode] = episode_reward
        makespan[episode] = env.step_count
        energy[episode] = sum(env.agents_energy_consumption)
            
        print(f"Episode {episode}, Total Reward = {episode_reward:.2f}", "jumlah step:", env.step_count, 
              "energy:", sum(env.agents_energy_consumption),
                "success:", success[episode], "success_energy:", success_energy[episode])
        # Save rewards to JSON
        combined_data = {
        "rewards": rewards,
        "makespan": makespan,
        "energy": energy,
         "success": success,
         "success_energy": success_energy,}
        
        # Write the combined dictionary to a single JSON file
        file_path = os.path.join(CURRENT_DIR, "Testing_DQN_yr3_1000ep_600_80max.json")
        with open(file_path, "w") as f:
            json.dump(combined_data, f, indent=4)


    env.close()
    return rewards

if __name__ == "__main__":
    all_rewards = run_env(num_episodes=200, render=False)
    print("Average Reward:", np.mean(list(all_rewards.values())))
