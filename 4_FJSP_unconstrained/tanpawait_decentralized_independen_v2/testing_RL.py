import gymnasium as gym
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from env_1 import FJSPEnv  
from RULED_BASED import MASKING_action
# Environment settings
RENDER_MODE = None
EPISODE_START =  600

STATE_DIM = 7
ACTION_DIM = 3
print(f"State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")

# Model directory
# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
PARENT_DIR = os.path.dirname(CURRENT_DIR)
print("PARENT_DIR:", PARENT_DIR)
MODEL_DIR = os.path.join(PARENT_DIR,'tanpawait_decentralized_independen_v2', "DQN_1")


class DDQN_model:
    def __init__(self, state_dim, action_dim, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir
        self.num_agents = 3
        self.dqn_network_dict={}
        self.replay_buffer_dict= {}
        for r in range(self.num_agents):
            self.dqn_network_dict[r] = self.create_dqn_network()


    def create_dqn_network(self):
        state_input = layers.Input(shape=(self.state_dim, ))
        
        # Apply Dense layers without activation and then apply LeakyReLU using TimeDistributed
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def load_models(self, episode):
        """Memuat bobot model DQN dan target DQN dari file .h5."""
        for r in range(self.num_agents):
            dqn_load_path = os.path.join(self.load_dir, f'dqn_agent_{r}_episode_{episode}.h5')
            print(f"Loading models from {dqn_load_path}")
            
            for path in [dqn_load_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
            
            self.dqn_network_dict[r].load_weights(dqn_load_path)
        print(f'Models loaded from episode {episode}')

        return self.dqn_network_dict

# Load models
loader = DDQN_model(state_dim=STATE_DIM, action_dim=ACTION_DIM, load_dir=MODEL_DIR)
dqn_network_dict = loader.load_models(episode=EPISODE_START)
for r in range(loader.num_agents):
    bias_output = dqn_network_dict[r].layers[-1].get_weights()[-1]
    print("bias_output:", bias_output)

@tf.function
def select_action_with_masking(r, state, action_mask_all):
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
    q_values = dqn_network_dict[r](state_tensor) 
    #tf.print("q_values:", q_values)
    q_values = q_values[0]  # shape (action_dim, )
   # tf.print("q_values:", q_values)
    action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)

    masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
   # tf.print("masked_q_values:", masked_q_values)
    actions = tf.argmax(masked_q_values)
    return actions

def normalize(state):
    state_mins = np.array([3, 1, 1, 0, 0, 0, 0,  ])
    state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, ])

    # Normalize each feature (column) using broadcasting.
    normalized_state = (state - state_mins) / (state_maxs - state_mins)
    return np.float32(normalized_state)

# Running environment
def run_env(num_episodes, render):
    env = FJSPEnv(window_size=1, num_agents=3, max_steps=1000, episode=1)
    num_episodes = 1000
    rewards = {}
    makespan = {}
    energy= {}
    for episode in range(1, num_episodes+1):
        all_states, info = env.reset(seed=1000+episode)
        if (episode-1) %1 == 0:
            episode_seed= episode-1
        env.conveyor.episode_seed= episode_seed
        episode_reward = 0
        done, truncated = False, False
        while not done:
            all_actions= None

            all_states_normalized = normalize(all_states) # Normalize the state
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            all_mask_actions = MASKING_action(all_states, env)
            all_actions = []
            for r in range(loader.num_agents):
                all_actions.append(select_action_with_masking(r, all_states_normalized[r], all_mask_actions[r]))


            all_next_states, all_rewards, done, truncated, info =  env.step(all_actions)

            #env.render()
            episode_reward +=  np.mean(all_rewards)
            all_states = all_next_states
            
            if env.FAILED_ACTION:
                print("FAILED ENV")
                break


        rewards[episode] = episode_reward
        makespan[episode] = env.step_count
        energy[episode] = sum(env.agents_energy_consumption)
        print(f"Episode {episode}, Total Reward = {episode_reward:.2f}", "jumlah step:", env.step_count, 
              "energy:", sum(env.agents_energy_consumption))
        # Save rewards to JSON
        combined_data = {
            "rewards": rewards,
            "makespan": makespan,
            "energy": energy
        }

        # Write the combined dictionary to a single JSON file
        file_path = os.path.join(CURRENT_DIR, "Testing_DQN_decent_independenv2_200ep.json")
        with open(file_path, "w") as f:
            json.dump(combined_data, f, indent=4)


    env.close()
    return rewards


if __name__ == "__main__":
    all_rewards = run_env(num_episodes=100, render=False)
    print("Average Reward:", np.mean(list(all_rewards.values())))
