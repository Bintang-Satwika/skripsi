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

from env_tanpawait import FJSPEnv  # Your flexible job shop environment
from RULED_BASED import MASKING_action, FCFS_action, RANDOM_action



class DDQN_model:
    def __init__(self,
                 buffer_length=500000,   # maximum number of transitions to store
                 batch_size=128,         # number of transitions per training batch
                 update_delay=1,
                 lr=0.0001,
                 tau=0.01,
                 gamma=0.98,
                 save_every_episode=20,
                 episode_load=0):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'DQN_tanpawait_3')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.num_agents = 3
        self.state_dim = 9
        self.action_dim = 3
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_delay = update_delay
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decay = 0.995
        # Use a deque for a buffer that stores individual (joint) transitions.
        self.replay_buffer = deque(maxlen=buffer_length)
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}
        self.episode_load= episode_load

        # Create Dense networks (shared among agents).
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        self.update_target_weights()

        # dir_location='model_HITL_Masking_16_381'
        # self.dqn_network, self.target_dqn_network = self.load_models(load_dir=os.path.join(self.current_dir, dir_location), 
        #                                                              episode=self.episode_load)

        bias_output = self.dqn_network.layers[-1].get_weights()[-1]
        print("bias_output:", bias_output)
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # replay_buffer_path = os.path.join(self.current_dir, dir_location, f'replay_buffer_episode_{self.episode_load}.pkl')
        
        # with open(replay_buffer_path , 'rb') as f:
        #     self.replay_buffer= pickle.load(f)
        print('len(self.replay_buffer):', len(self.replay_buffer))
    
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
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update_RL_memory(self, state, action, reward, next_state, done, mask_actions, mask_actions_next):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.replay_buffer.append((state, action, reward, next_state, done, mask_actions, mask_actions_next))

    def take_RL_minibatch(self):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        # Unpack and stack the multi-agent data from each tuple
        mb_states = tf.convert_to_tensor(np.stack([data[0] for data in minibatch]), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.stack([data[1] for data in minibatch]), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.stack([data[2] for data in minibatch]), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.stack([data[3] for data in minibatch]), dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(np.stack([data[4] for data in minibatch]), dtype=tf.float32)
        mb_action_mask= tf.convert_to_tensor(np.stack([data[5] for data in minibatch]), dtype=tf.float32)
        mb_action_mask_next = tf.convert_to_tensor(np.stack([data[6] for data in minibatch]), dtype=tf.float32)
        # print("mb_states: ", mb_states.shape)
        # print("mb_actions: ", mb_actions.shape)
        # print("mb_rewards: ", mb_rewards.shape)
        # print("mb_next_states: ", mb_next_states.shape)
        # print("mb_dones: ", mb_dones.shape)
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones, mb_action_mask, mb_action_mask_next

    @tf.function
    def train_dqn(self):
        # Assume take_RL_minibatch now returns an additional element: mb_action_mask_next
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones, mb_action_mask, mb_action_mask_next = self.take_RL_minibatch()
        # mb_states and mb_next_states: (batch, num_agents, state_dim)
        # mb_actions, mb_rewards, mb_dones: (batch, num_agents)
        # mb_action_mask_next: (batch, num_agents, action_dim)

        mb_actions = tf.cast(mb_actions, tf.int32)
        mb_action_mask_next = tf.cast(mb_action_mask_next, tf.bool)
        #tf.print("mb_action_mask_next: ", mb_action_mask_next.shape)
        
        with tf.GradientTape() as tape:
            # --- Current-State Q-Values (for loss computation) ---
            # Compute Q-values for current states using the online network
            mb_Q_values = self.dqn_network(mb_states, training=True)  # shape (batch, num_agents, action_dim)
            # Select Q-values corresponding to the taken actions
            mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), axis=2)
            
            # --- Next-State Q-Values (for target) with masking ---
            # Use the target network to compute Q-values for the next states
            mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)  # shape (batch, num_agents, action_dim)
            #tf.print("mb_target_Q_values_next: ", mb_target_Q_values_next.shape)
            # Mask out invalid actions by replacing their Q-values with -infinity
            masked_mb_target_Q_values_next = tf.where(mb_action_mask_next,
                                                    mb_target_Q_values_next,
                                                    tf.fill(tf.shape(mb_target_Q_values_next), -np.inf))
            # Take the maximum Q-value over valid actions for each agent
            max_mb_target_Q_values_next = tf.reduce_max(masked_mb_target_Q_values_next, axis=2)  # shape (batch, num_agents)
            
            # --- Compute the target Q-value using the Bellman equation ---
            # y = reward + gamma * max_next_Q_value * (1 - done)
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * max_mb_target_Q_values_next
            
            # Compute the loss as the mean squared error over all agents and batches
            loss = tf.reduce_mean(tf.square(y - mb_Q_values))

        # Compute gradients and update the online network's weights
        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        del tape


        # Periodically update target network (soft update)
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights()

    def normalize(self, state):
        state_mins = np.array([3, 1, 1, 0, 0, 0, 0, 0, 0 ])
        state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, 3, 3])

        # Normalize each feature (column) using broadcasting.
        normalized_state = (state - state_mins) / (state_maxs - state_mins)
        return np.float32(normalized_state)


if __name__ == "__main__":
    DDQN = DDQN_model()
    env = FJSPEnv(window_size=1, num_agents=3, max_steps=1000, episode=DDQN.episode_load)
    num_agents = env.num_agents
    num_episodes = 500

    for episode in tqdm(range(DDQN.episode_load+1, num_episodes + 1)):
        state, info = env.reset(seed=episode)
        print("Initial state:", state)
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
                joint_actions = RANDOM_action(state, env)  
            else:
                joint_actions= DDQN.select_action_with_masking(state_normalized, mask_actions)

            if None in joint_actions:
                print("FAILED ACTION: ", joint_actions)
                break

            next_state, reward, done, truncated, info = env.step(joint_actions)
            reward = np.mean(reward)
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
                #print("mask_actions_next: ", mask_actions_next)
                DDQN.update_RL_memory(state_normalized, joint_actions, np.repeat(reward, num_agents), next_state_normalized, np.repeat(done, num_agents), mask_actions, mask_actions_next)

            
            if len(DDQN.replay_buffer) >= DDQN.batch_size:
                DDQN.train_dqn()

            state = next_state
            #--------------------------------------------------------------------------------------------------------

        if env.FAILED_ACTION or None in joint_actions:
            print("state: ", state)
            print("mask_actions: ", mask_actions)
            print("joint_actions: ", joint_actions)
            print("FAILED EPISODE")
            break
        DDQN.update_epsilon()

        DDQN.cumulative_reward_episode[episode] = reward_episode
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed),
              "energy consumption:", sum(env.agents_energy_consumption),
              "epsilon:", DDQN.epsilon
              )
