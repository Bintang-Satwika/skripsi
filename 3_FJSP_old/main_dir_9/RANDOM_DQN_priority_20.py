''' hanya beda masking_action dari fcfs
    dan parameter fungsi reward penalty  == wait
 '''
import os
import random
import pickle
import json
from collections import deque
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from env_2 import FJSPEnv  # Your flexible job shop environment
from RULED_BASED import MASKING_action, HITL_action, FCFS_action, RANDOM_action

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-5, epsilon=1e-6):
        """
        capacity: Maximum number of transitions to store.
        alpha:      Exponent for priorities (0: no prioritization; 1: full prioritization).
        beta:       Initial exponent for importance-sampling.
        beta_increment: Increment for beta after each sampling.
        epsilon:    Small constant to avoid zero priority.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # NEW: Use deque for the replay buffer and priorities.
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def store(self, state, action, reward, next_state, done, mask_actions, mask_actions_next):
        """
        Store a new transition into the replay buffer.
        Each argument should have shapes:
          state, next_state: (num_agents, state_dim)
          action, reward, done: (num_agents,) or scalar
          mask_actions, mask_actions_next: (num_agents, action_dim) booleans
        """
        # NEW: When storing, append to the deque.
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0
        self.buffer.append((state, action, reward, next_state, done, mask_actions, mask_actions_next))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """
        Sample a minibatch of transitions.
        Returns:
          minibatch: list of transitions.
          weights:   importance-sampling weights, shape (batch_size,).
          indices:   indices of sampled transitions, shape (batch_size,).
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer!")

        valid_size = len(self.buffer)
        # Convert deque of priorities to a NumPy array.
        priorities_np = np.array(self.priorities, dtype=np.float32)
        # NEW: Add epsilon to avoid zero, then raise to power alpha.
        scaled_priorities = (priorities_np + self.epsilon) ** self.alpha
        sum_scaled = np.sum(scaled_priorities)
        if sum_scaled == 0:
            probs = np.ones(valid_size) / valid_size
        else:
            probs = scaled_priorities / sum_scaled

        # Sample indices according to the computed probabilities.
        indices = np.random.choice(valid_size, batch_size, p=probs)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Compute importance-sampling weights.
        N = valid_size
        p = probs[indices]
        weights = (N * p) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability.

        minibatch = [self.buffer[idx] for idx in indices]
        return minibatch, weights, indices

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities for the sampled transitions using new TD errors.
        td_errors: array-like of shape (batch_size,).
        """
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + self.epsilon


class DDQN_model:
    def __init__(self,
                 buffer_length=500000,   # maximum number of transitions to store
                 batch_size=128,         # number of transitions per training batch
                 update_delay=100,
                 lr=0.0000625,
                 tau=0.001,
                 gamma=0.98,
                 save_every_episode=20,
                 episode_load=0):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'model_RANDOM_priority_20')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.num_agents = 3
        self.state_dim = 14
        self.action_dim = 5
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_delay = update_delay
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decay = 0.995
     
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}

        self.episode_load= episode_load

        # Create Dense networks (shared among agents).
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        # self.dqn_network, self.target_dqn_network = self.load_models(load_dir=os.path.join(self.current_dir, 'model_HITL_Masking_21_400_261ep'),
        #                                                              episode=self.episode_load)
        bias_output = self.dqn_network.layers[-1].get_weights()[-1]
        print("bias_output:", bias_output)
        #self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_length,
            alpha=0.6,
            beta=0.4,
            beta_increment=1e-5,
            epsilon=1e-4
        )

    
    def load_models(self,load_dir, episode):
        """Memuat bobot model DQN dan target DQN dari file .h5."""
        dqn_load_path = os.path.join(load_dir, f'dqn_episode_{episode}.h5')
        target_dqn_load_path = os.path.join(load_dir, f'target_dqn_episode_{episode}.h5')

        for path in [dqn_load_path, target_dqn_load_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.dqn_network.load_weights(dqn_load_path)
        self.target_dqn_network.load_weights(target_dqn_load_path)
        print(f'Models loaded from episode {episode}')

        return self.dqn_network, self.target_dqn_network

    def save_models(self, episode):
        if episode % self.save_every_episode == 0:
            dqn_save_path = os.path.join(self.save_dir, f'dqn_episode_{episode}.h5')
            target_dqn_save_path = os.path.join(self.save_dir, f'target_dqn_episode_{episode}.h5')
            self.dqn_network.save_weights(dqn_save_path)
            self.target_dqn_network.save_weights(target_dqn_save_path)
            print(f'Models saved at episode {episode}')

    def save_replay_buffer_and_rewards(self, episode):
        # Save replay buffer every few episodes.
        if episode % self.save_every_episode == 0:
            replay_filename = os.path.join(self.save_dir, f'replay_buffer_episode_{episode}.pkl')
            with open(replay_filename, 'wb') as f:
                pickle.dump(self.replay_buffer, f)


        rewards_filename = os.path.join(self.save_dir, 'cumulative_rewards.json')
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
        # For 3 agents: input shape becomes (num_agents, state_dim)
        state_input = layers.Input(shape=(self.num_agents, self.state_dim))
        
        # Apply Dense layers without activation and then apply LeakyReLU using TimeDistributed
        x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(state_input)
        x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
        X = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
        output = layers.TimeDistributed(layers.Dense(self.action_dim, activation='linear'))(x)
        
        model = models.Model(inputs=state_input, outputs=output)
        return model


    def update_target_weights(self):
        # Soft update: target = tau * main + (1-tau) * target.
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)


    def select_action_with_masking(self, state, action_mask_all):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.dqn_network(state_tensor)  # shape (1, num_agents, action_dim)
    
        q_values = q_values[0]  # shape (num_agents, action_dim)

        action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)

        masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
        actions = tf.argmax(masked_q_values, axis=1)
        return actions.numpy()
    
    def update_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

    def update_RL_memory(self, state, action, reward, next_state, done, mask_actions, mask_actions_next):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.replay_buffer.store(state, action, reward, next_state, done, mask_actions, mask_actions_next)

    def take_RL_minibatch(self):
        """
        Sample from the prioritized replay buffer.
        Returns the batch in TF tensors, plus (weights, indices).
        """
        minibatch, weights, indices = self.replay_buffer.sample(self.batch_size)

        # Unpack transitions
        states      = np.stack([mb[0] for mb in minibatch])  # shape (batch, num_agents, state_dim)
        actions     = np.stack([mb[1] for mb in minibatch])  # shape (batch, num_agents)
        rewards     = np.stack([mb[2] for mb in minibatch])  # shape (batch, num_agents)
        next_states = np.stack([mb[3] for mb in minibatch])  # shape (batch, num_agents, state_dim)
        dones       = np.stack([mb[4] for mb in minibatch])  # shape (batch, num_agents)
        mask_a      = np.stack([mb[5] for mb in minibatch])  # shape (batch, num_agents, action_dim)
        mask_a_next = np.stack([mb[6] for mb in minibatch])  # shape (batch, num_agents, action_dim)

        # Convert to tf tensors
        mb_states      = tf.convert_to_tensor(states, dtype=tf.float32)
        mb_actions     = tf.convert_to_tensor(actions, dtype=tf.int32)
        mb_rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        mb_dones       = tf.convert_to_tensor(dones, dtype=tf.float32)
        mb_action_mask = tf.convert_to_tensor(mask_a, dtype=tf.bool)
        mb_action_mask_next = tf.convert_to_tensor(mask_a_next, dtype=tf.bool)

        tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        return (mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones,
                mb_action_mask, mb_action_mask_next, tf_weights, indices)

    def train_double_dqn(self):
        # Assume take_RL_minibatch now returns an additional element: mb_action_mask_next
        (mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones,
         mb_action_mask, mb_action_mask_next, weights, indices) = self.take_RL_minibatch()

        mb_actions = tf.cast(mb_actions, tf.int32)
        mb_action_mask_next  = tf.cast(mb_action_mask_next , tf.bool)
        
        with tf.GradientTape() as tape:
            # --- Next-State Q-Values (for target) with masking ---
            # Get Q-values for the next states: shape (batch, num_agents, action_dim)
            mb_Q_values_next = self.dqn_network(mb_next_states, training=False)
            # Replace invalid actions' Q-values with -infinity so they won't be selected
            masked_mb_Q_values_next = tf.where(mb_action_mask_next,
                                            mb_Q_values_next,
                                            tf.fill(tf.shape(mb_Q_values_next), -np.inf))
            # Select the best valid action for each agent
            mb_actions_next = tf.argmax(masked_mb_Q_values_next, axis=2)  # shape (batch, num_agents)
            
            # --- Current-State Q-Values (for loss computation) ---
            mb_Q_values = self.dqn_network(mb_states, training=True)  # shape (batch, num_agents, action_dim)
            # Select the Q-values corresponding to the taken actions using one-hot encoding
            mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), axis=2)
            
            # --- Target Q-Values for next states using target network ---
            mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)
            # For the target network, we use the action indices selected from the masked Q-values
            mb_target_Q_values_next = tf.reduce_sum(mb_target_Q_values_next * tf.one_hot(mb_actions_next, self.action_dim), axis=2)
            
            # --- Compute the target Q-value (Bellman equation) ---
            # y = reward + gamma * target_q_value * (1 - done)
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * mb_target_Q_values_next
            
            # 5) TD error
            td_errors = y - mb_Q_values

            # For a multi-agent transition, average the TD error across agents
            # so we get one "priority" per batch element:
            td_errors_mean = tf.reduce_mean(td_errors, axis=1)  # shape: (batch,)

            # 6) Weighted MSE loss
            #    We'll compute MSE per sample, average across agents, then multiply by PER weights.
            #    shape: (batch,)
            mse_per_sample = tf.reduce_mean(tf.square(td_errors), axis=1)
            weighted_mse = weights * mse_per_sample
            loss = tf.reduce_mean(weighted_mse)

        # 7) Backpropagation
        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        del tape

        # 8) Update priorities in the buffer
        abs_td_errors = tf.abs(td_errors_mean).numpy()
        self.replay_buffer.update_priorities(indices, abs_td_errors)

        # 9) update target network (soft update)
        self.update_target_weights()

    def normalize(self, state):
        state_mins = np.array([3,  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2])

        # Normalize each feature (column) using broadcasting.
        normalized_state = (state - state_mins) / (state_maxs - state_mins)
        return np.float32(normalized_state)
    


if __name__ == "__main__":
    DDQN = DDQN_model()
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000, episode=DDQN.episode_load)
    num_agents = env.num_agents
    num_episodes = 1000
    for episode in tqdm(range(DDQN.episode_load+1, num_episodes + 1)):
        state, info = env.reset(seed=episode)
        if (episode-1) %1 == 0:
            episode_seed= episode-1
        env.conveyor.episode_seed= episode_seed
        reward_episode = 0
        done = False
        truncated = False
        human_actions=None
        print("\nEpisode:", episode)

        # For joint transitions, we initialize a single episode buffer.
        
        while not done:
            joint_actions= None
            DDQN.iterasi += 1
            state_normalized = DDQN.normalize(state) # Normalize the state
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break


            mask_actions = MASKING_action(state, env)
            if np.random.rand() < DDQN.epsilon:
                if episode % 3 == 1:
                    joint_actions, _ = HITL_action(state, env)  # Episode 1, 4, 7, ...
                elif episode % 3 == 2:
                    joint_actions = FCFS_action(state, env)     # Episode 2, 5, 8, ...
                elif episode % 3 == 0:
                    joint_actions = RANDOM_action(state, env)   # Episode 3, 6, 9, ...
                else:
                    print("ERROR RANDOM FAILED ACTION: ", joint_actions)
                    break
            else:
                joint_actions= DDQN.select_action_with_masking(state_normalized, mask_actions)


            if None in joint_actions:
                print("FAILED ACTION: ", joint_actions)
                break

            next_state, reward, done, truncated, info = env.step(joint_actions)
            
            next_state_normalized = DDQN.normalize(next_state)

            if env.FAILED_ACTION:
                print("FAILED ENV")
                print("failed RL")
                print("state: ", state)
                print("mask_actions: ", mask_actions)
                print("joint_actions: ", joint_actions)
                print("next_state: ", next_state)
                break

            done = done or truncated
            reward_episode += reward

            if not np.array_equal(next_state, state):
                mask_actions_next = MASKING_action(next_state, env)
                DDQN.update_RL_memory(state_normalized, joint_actions, reward, next_state_normalized, np.repeat(done, num_agents), mask_actions, mask_actions_next)
            
            if len(DDQN.replay_buffer) >= DDQN.batch_size and DDQN.iterasi % 4 == 0:
                DDQN.train_double_dqn()

            state = next_state
            #--------------------------------------------------------------------------------------------------------

        if env.FAILED_ACTION or None in joint_actions:
            print("FAILED EPISODE")
            break
        
        
        DDQN.update_epsilon()
        DDQN.cumulative_reward_episode[episode] = np.mean(reward_episode)
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed),
              "energy consumption:", sum(env.agents_energy_consumption),
              )
