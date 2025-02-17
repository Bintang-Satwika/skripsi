import os
import random
import pickle
import json
from collections import deque
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from env_5c_3 import FJSPEnv  # Your environment


class DDQN_model:
    def __init__(self,
                 buffer_length=1000,  # maximum number of episodes stored
                 batch_size=64,         # number of episodes (i.e. traces) per training batch
                 update_delay=2,
                 lr=0.0001,
                 tau=0.005,
                 gamma=0.98,
                 trace_length=4,        # τ: length of each trace
                 save_every_episode=20):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'saved_CERT_2')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.state_dim = 14
        self.action_dim = 5
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.epsilon = 1.0
        self.epsilon_decay = 0.92
        self.update_delay = update_delay
        self.batch_size = batch_size
        self.trace_length = trace_length

        # Create DRQN networks with LSTM to process sequences of states.
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        self.update_target_weights()

        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        # CERT_buffer now stores complete episodes.
        self.CERT_buffer = deque(maxlen=int(buffer_length))
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}

    def save_models(self, episode):
        dqn_save_path = os.path.join(self.save_dir, f'dqn_episode_{episode}.h5')
        target_dqn_save_path = os.path.join(self.save_dir, f'target_dqn_episode_{episode}.h5')
        self.dqn_network.save_weights(dqn_save_path)
        self.target_dqn_network.save_weights(target_dqn_save_path)
        print(f'Models saved at episode {episode}')

    def save_replay_buffer_and_rewards(self, episode):
        # Save CERT buffer every few episodes.
        if episode % self.save_every_episode == 0:
            replay_filename = os.path.join(self.save_dir, f'CERT_buffer_episode_{episode}.pkl')
            with open(replay_filename, 'wb') as f:
                pickle.dump(self.CERT_buffer, f)

        rewards_filename = os.path.join(self.save_dir, 'A_cumulative_rewards.json')
        if os.path.exists(rewards_filename):
            with open(rewards_filename, 'r') as f:
                existing_rewards = json.load(f)
        else:
            existing_rewards = {}
        reward_value = self.cumulative_reward_episode[episode]
        if isinstance(reward_value, (np.ndarray,)):
            reward_value = reward_value.tolist()
        existing_rewards[str(episode)] = reward_value
        with open(rewards_filename, 'w') as f:
            json.dump(existing_rewards, f, indent=4)
        print(f'Cumulative rewards saved to {rewards_filename}')

    def create_dqn_network(self):
        # Revised DRQN: input shape is (trace_length, state_dim)
        state_input = layers.Input(shape=(self.trace_length, self.state_dim))
        # Use an LSTM layer to capture sequential correlations.
        x = layers.LSTM(64, return_sequences=False)(state_input)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def update_target_weights(self):
        # Soft update: target = tau * main + (1 - tau) * target.
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)

    def select_action_with_masking(self, state, action_mask):
        # 'state' here is a single state vector (shape = (state_dim,))
        if np.random.rand() < self.epsilon:
            valid_actions = np.where(action_mask)[0]
            return int(random.choice(valid_actions))
        else:
            # Expand state to a sequence of length=trace_length by repeating the state.
            # (This is a simple trick for action selection when no history is available.)
            state_seq = np.tile(state, (self.trace_length, 1))
            state_tensor = tf.convert_to_tensor([state_seq], dtype=tf.float32)  # shape: (1, trace_length, state_dim)
            q_values = self.dqn_network(state_tensor)
            action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
            masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
            return int(tf.argmax(masked_q_values, axis=1)[0])
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def take_CERT_minibatch(self):
        # Sample a concurrent minibatch of traces from the CERT buffer.
        if len(self.CERT_buffer) < self.batch_size:
            return None  # not enough episodes yet
        sampled_episodes = random.sample(list(self.CERT_buffer), self.batch_size)
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = [], [], [], [], []
        # Each episode is a list of traces per agent; assume all agents have the same number T of transitions.
        for episode in sampled_episodes:
            # Each 'episode' is a list of length=num_agents, where each element is a list of transitions.
            T = len(episode[0])
            if T < self.trace_length:
                continue  # skip episodes that are too short
            t0 = random.randint(0, T - self.trace_length)
            # For each agent in the episode, extract a trace from t0 to t0 + trace_length
            for agent_trace in episode:
                trace = agent_trace[t0 : t0 + self.trace_length]
                states = [transition[0] for transition in trace]
                actions = [transition[1] for transition in trace]
                rewards = [transition[2] for transition in trace]
                next_states = [transition[3] for transition in trace]
                dones = [transition[4] for transition in trace]
                mb_states.append(states)
                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_next_states.append(next_states)
                mb_dones.append(dones)
        if len(mb_states) == 0:
            return None
        # Convert lists to tensors.
        mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)          # shape: (batch, trace_length, state_dim)
        mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.int32)             # shape: (batch, trace_length)
        mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)           # shape: (batch, trace_length)
        mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)    # shape: (batch, trace_length, state_dim)
        mb_dones = tf.convert_to_tensor(mb_dones, dtype=tf.float32)               # shape: (batch, trace_length)
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones

    def train_double_dqn(self):
        minibatch = self.take_CERT_minibatch()
        if minibatch is None:
            return  # not enough data to train
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = minibatch
        # For training, we use the final timestep in each trace.
        last_states = mb_states[:, -1, :]       # shape: (batch, state_dim)
        last_next_states = mb_next_states[:, -1, :]
        last_actions = mb_actions[:, -1]          # shape: (batch,)
        last_rewards = mb_rewards[:, -1]          # shape: (batch,)
        last_dones = mb_dones[:, -1]              # shape: (batch,)

        with tf.GradientTape() as tape:
            # Get Q-values for current sequences (output from last timestep).
            Q_values = self.dqn_network(mb_states, training=True)  # shape: (batch, action_dim)
            # Q-value for the taken actions:
            Q_taken = tf.reduce_sum(Q_values * tf.one_hot(last_actions, self.action_dim), axis=1)
            # Use main network to select next actions:
            Q_next_main = self.dqn_network(mb_next_states, training=False)
            actions_next = tf.argmax(Q_next_main, axis=1)
            # Evaluate next Q-values using target network:
            Q_next_target = self.target_dqn_network(mb_next_states, training=False)
            Q_next_target_taken = tf.reduce_sum(Q_next_target * tf.one_hot(actions_next, self.action_dim), axis=1)
            # Compute target value (for final timestep in the trace):
            y = last_rewards + (1.0 - last_dones) * self.gamma * Q_next_target_taken
            loss = tf.reduce_mean(tf.square(y - Q_taken))
        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights()
        # Optionally, print loss or other diagnostics:
        # print(f"Loss: {loss.numpy()}")

# Helper function for masking actions remains largely unchanged.
def masking_action(states, env):
    mask_actions = []
    for i, state in enumerate(states):
        is_agent_working = (state[env.state_operation_now_location] != 0)
        is_status_idle = (state[env.state_status_location_all[i]] == 0) 
        is_status_accept = (state[env.state_status_location_all[i]] == 1)
        is_pick_job_window_yr_1 = (state[env.state_pick_job_window_location] == 1)
        is_pick_job_window_yr_2 = (state[env.state_pick_job_window_location] == 2)
        is_job_in_capability_yr = (state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location])
        is_job_in_capability_yr_1 = (state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location])
        is_job_in_capability_yr_2 = (state[env.state_first_job_operation_location[2]] in state[env.state_operation_capability_location])
        # Set up booleans for each action.
        accept_action = False
        wait_yr_1_action = False
        wait_yr_2_action = False
        decline_action = False
        continue_action = False
        if is_status_accept:
            accept_action = True
        elif not is_agent_working:
            if is_status_idle and is_pick_job_window_yr_1 and is_job_in_capability_yr:
                accept_action = True
                decline_action = True
                continue_action = True
            if is_status_idle and is_job_in_capability_yr_2:
                wait_yr_2_action = True
                decline_action = True
            if is_status_idle and is_job_in_capability_yr_1:
                wait_yr_1_action = True
                decline_action = True
            continue_action = True
        elif is_agent_working:
            if not is_status_idle:
                continue_action = True
        mask_actions.append([accept_action, wait_yr_1_action, wait_yr_2_action, decline_action, continue_action])
    return mask_actions


if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000)
    DDQN = DDQN_model()
    num_agents = env.num_agents
    num_episodes = 500

    for episode in tqdm(range(1, num_episodes + 1)):
        state, info = env.reset(seed=episode)
        reward_episode = 0
        done = False
        truncated = False
        print("\nEpisode:", episode)

        # Initialize an episode buffer for each agent.
        episode_buffer = [[] for _ in range(num_agents)]
        
        while not done:
            DDQN.iterasi += 1
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            mask_actions = masking_action(state, env)
            actions = []
            for single_state, mask in zip(state, mask_actions):
                action = DDQN.select_action_with_masking(single_state, mask)
                actions.append(action)
            actions = np.array(actions)
            if None in actions:
                print("FAILED ACTION: ", actions)
                break

            next_state, reward, done, truncated, info = env.step(actions)
            done = done or truncated
            reward_episode += reward

            if env.FAILED_ACTION:
                print("FAILED ENV")
                break

            # Only store transition if next_state is different from state.
            if not np.array_equal(next_state, state):
                for r in range(num_agents):
                    transition = (state[r], actions[r], reward[r], next_state[r], done)
                    episode_buffer[r].append(transition)
            state = next_state

        if env.FAILED_ACTION:
            print("FAILED ENV, skipping episode storage.")
            break

        # After the episode, add the episode_buffer (which is a list per agent) to the CERT buffer.
        DDQN.CERT_buffer.append(episode_buffer)
        DDQN.update_epsilon()
        DDQN.cumulative_reward_episode[episode] = reward_episode
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed))
