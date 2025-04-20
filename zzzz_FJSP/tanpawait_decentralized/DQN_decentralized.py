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
                 save_every_episode=50,
                 episode_load=0):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'DQN_decentralized_1')
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
        self.epsilon_decay = 0.99
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}
        self.episode_load= episode_load

        # Create Dense networks (shared among agents).
        self.dqn_network_dict={}
        self.target_dqn_network_dict={}
        self.replay_buffer_dict= {}
        for r in range(self.num_agents):
            self.dqn_network_dict[r] = self.create_dqn_network()
            self.target_dqn_network_dict[r] = self.create_dqn_network()
            self.target_dqn_network_dict[r].set_weights(self.dqn_network_dict[r].get_weights())
            # Use a deque for a buffer that stores individual (joint) transitions.
            self.replay_buffer_dict[r] = deque(maxlen=buffer_length)
        

        # dir_location='model_HITL_Masking_16_381'
        # self.dqn_network, self.target_dqn_network = self.load_models(load_dir=os.path.join(self.current_dir, dir_location), 
        #                                                              episode=self.episode_load)
        self.dqn_optimizer_dict = {}
        for r in range(self.num_agents):
            bias_output = self.dqn_network_dict[r].layers[-1].get_weights()[-1]
            print("bias_output:", bias_output)
            self.dqn_optimizer_dict[r] = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # replay_buffer_path = os.path.join(self.current_dir, dir_location, f'replay_buffer_episode_{self.episode_load}.pkl')
        
        # with open(replay_buffer_path , 'rb') as f:
        #     self.replay_buffer= pickle.load(f)
        print('len(self.replay_buffer):', len(self.replay_buffer_dict[0]))
    
    # def load_models(self,load_dir, episode):
    #     """Memuat bobot model DQN dan target DQN dari file .h5."""
    #     dqn_load_path = os.path.join(load_dir, f'dqn_episode_{episode}.h5')
    #     target_dqn_load_path = os.path.join(load_dir, f'target_dqn_episode_{episode}.h5')

    #     for path in [dqn_load_path, target_dqn_load_path]:
    #         if not os.path.exists(path):
    #             raise FileNotFoundError(f"Model file not found: {path}")

    #     self.dqn_network.load_weights(dqn_load_path)
    #     self.target_dqn_network.load_weights(target_dqn_load_path)
    #     print(f'Models loaded from episode {episode}')

    #     return self.dqn_network, self.target_dqn_network

    def save_models(self, episode):
        if episode % self.save_every_episode == 0:
            for r in range(self.num_agents):
                dqn_save_path = os.path.join(self.save_dir, f'dqn_agent_{r}_episode_{episode}.h5')
                target_dqn_save_path = os.path.join(self.save_dir, f'target_dqn_agent_{r}_episode_{episode}.h5')
                self.dqn_network_dict[r].save_weights(dqn_save_path)
                self.target_dqn_network_dict[r].save_weights(target_dqn_save_path)

            print(f'Models saved at episode {episode}')

    def save_replay_buffer_and_rewards(self, episode):
        if episode % self.save_every_episode == 0:
            for r in range(self.num_agents):
                replay_filename = os.path.join(self.save_dir, f'replay_buffer_agent_{r}_episode_{episode}.pkl')
                with open(replay_filename, 'wb') as f:
                    pickle.dump(self.replay_buffer_dict[r], f)
            
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
        state_input = layers.Input(shape=(self.state_dim, ))
        
        # Apply Dense layers without activation and then apply LeakyReLU using TimeDistributed
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(128, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        
        model = models.Model(inputs=state_input, outputs=output)
        return model


    def update_target_weights(self, r):
        # Soft update: target = tau * main + (1-tau) * target.
        for main_var, target_var in zip(self.dqn_network_dict[r].trainable_variables,
                                        self.target_dqn_network_dict[r].trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)



    def select_action_with_masking(self, r, state, action_mask_all):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.dqn_network_dict[r](state_tensor) 
        q_values = q_values[0]  # shape (action_dim, )

        action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)

        masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
        actions = tf.argmax(masked_q_values)
        return actions.numpy()
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update_RL_memory(self, r, states, actions, rewards, next_states, done, mask_actions, mask_actions_next):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.replay_buffer_dict[r].append((states, actions, rewards, next_states, done, mask_actions, mask_actions_next))
         

    def take_RL_minibatch(self, r):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.replay_buffer_dict[r], self.batch_size)
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_done,  mb_action_mask, mb_action_mask_next = zip(*minibatch)
        mb_states = tf.convert_to_tensor(np.array(mb_states), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.array(mb_actions), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.array(mb_rewards), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.array(mb_next_states), dtype=tf.float32)
        mb_done = tf.convert_to_tensor(np.array(mb_done), dtype=tf.float32)
        mb_action_mask = tf.convert_to_tensor(np.array(mb_action_mask), dtype=tf.float32)
        mb_action_mask_next = tf.convert_to_tensor(np.array(mb_action_mask_next), dtype=tf.float32)
        # print("mb_states: ", np.shape(mb_states))
        # print("mb_actions: ", np.shape(mb_actions))
        # print("mb_rewards: ", np.shape(mb_rewards))
        # print("mb_next_states: ", np.shape(mb_next_states))
        # print("mb_done: ", np.shape(mb_done))
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_done, mb_action_mask, mb_action_mask_next

    # @tf.function
    # def train_dqn(self, r):
    #     # Assume take_RL_minibatch now returns an additional element: mb_action_mask_next
    #     mb_states, mb_actions, mb_rewards, mb_next_states, mb_done, mb_action_mask, mb_action_mask_next = self.take_RL_minibatch(r=r)
    #     mb_actions = tf.cast(mb_actions, tf.int32)
    #     mb_action_mask_next = tf.cast(mb_action_mask_next, tf.bool)
        
    #     with tf.GradientTape() as tape:
    #         # --- Current-State Q-Values (for loss computation) ---
    #         # Compute Q-values for current states using the online network
    #         mb_Q_values = self.dqn_network_dict[r](mb_states, training=True)  # shape (batch, num_agents, action_dim)
    #         # Select Q-values corresponding to the taken actions
    #         mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), axis=1)
            
    #         # --- Next-State Q-Values (for target) with masking ---
    #         # Use the target network to compute Q-values for the next states
    #         mb_target_Q_values_next = self.target_dqn_network_dict[r](mb_next_states, training=False)  # shape (batch, num_agents, action_dim)
    #         # Mask out invalid actions by replacing their Q-values with -infinity
    #         masked_mb_target_Q_values_next = tf.where(mb_action_mask_next,
    #                                                 mb_target_Q_values_next,
    #                                                 tf.fill(tf.shape(mb_target_Q_values_next), -np.inf))
    #         # Take the maximum Q-value over valid actions for each agent
    #         max_mb_target_Q_values_next = tf.reduce_max(masked_mb_target_Q_values_next, axis=1)  # shape (batch, num_agents)
            
    #         # --- Compute the target Q-value using the Bellman equation ---
    #         # y = reward + gamma * max_next_Q_value * (1 - done)
    #         y = mb_rewards + (1.0 - mb_done) * self.gamma * max_mb_target_Q_values_next
            
    #         # Compute the loss as the mean squared error over all agents and batches
    #         loss = tf.reduce_mean(tf.square(y - mb_Q_values))

    #     # Compute gradients and update the online network's weights
    #     grads = tape.gradient(loss, self.dqn_network_dict[r].trainable_variables)
    #     self.dqn_optimizer_dict[r].apply_gradients(zip(grads, self.dqn_network_dict[r].trainable_variables))
    #     del tape


    #     # Periodically update target network (soft update)
    #     if self.iterasi % self.update_delay == 0:
    #         self.update_target_weights(r=r)
    def train_dqn(self, r):
        """Main training loop for agent `r`. This runs outside @tf.function."""
        # Sample minibatch
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_done, mb_action_mask, mb_action_mask_next = self.take_RL_minibatch(r)

        # Convert action to int (for one-hot later)
        mb_actions = tf.cast(mb_actions, tf.int32)
        mb_action_mask_next = tf.cast(mb_action_mask_next, tf.bool)

        # Compute gradients and loss inside a @tf.function
        loss, grads = self._compute_gradients(mb_states, mb_actions, mb_rewards, mb_next_states,
                                            mb_done, mb_action_mask_next, r)

        # Apply gradients outside @tf.function (safe)
        self.dqn_optimizer_dict[r].apply_gradients(zip(grads, self.dqn_network_dict[r].trainable_variables))
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights(r=r)

    @tf.function
    def _compute_gradients(self, mb_states, mb_actions, mb_rewards, mb_next_states,
                        mb_done, mb_action_mask_next, r):
        """Runs inside @tf.function and only computes gradients."""
        with tf.GradientTape() as tape:
            # Q-values from online net
            mb_Q_values = self.dqn_network_dict[r](mb_states, training=True)
            mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), axis=1)

            # Q-values from target net (next states)
            mb_target_Q_values_next = self.target_dqn_network_dict[r](mb_next_states, training=False)
            neg_inf = tf.constant(-1e9, dtype=tf.float32)
            masked_q_next = tf.where(mb_action_mask_next,
                                    mb_target_Q_values_next,
                                    tf.fill(tf.shape(mb_target_Q_values_next), neg_inf))

            max_q_next = tf.reduce_max(masked_q_next, axis=1)
            y = mb_rewards + (1.0 - mb_done) * self.gamma * max_q_next

            # Compute loss
            loss = tf.reduce_mean(tf.square(y - mb_Q_values))

        grads = tape.gradient(loss, self.dqn_network_dict[r].trainable_variables)
        return loss, grads

    def normalize(self, state):
        state_mins = np.array([3, 1, 1, 0, 0, 0, 0, 0, 0 ])
        state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, 3, 3])

        # Normalize each feature (column) using broadcasting.
        normalized_state = (state - state_mins) / (state_maxs - state_mins)
        return np.float32(normalized_state)


if __name__ == "__main__":
    DDQN = DDQN_model()
    env = FJSPEnv(window_size=1, num_agents=3, max_steps=2000, episode=DDQN.episode_load)
    num_agents = env.num_agents
    num_episodes = 500

    for episode in tqdm(range(DDQN.episode_load+1, num_episodes + 1)):
        all_states, info = env.reset(seed=episode)
        if (episode-1) %1 == 0:
            episode_seed= episode-1
        env.conveyor.episode_seed= episode_seed
        reward_episode = [0] * num_agents
        done = False
        truncated = False
        print("\nEpisode:", episode)

        # For joint transitions, we initialize a single episode buffer.
        
        while not done:
            all_actions= None
            DDQN.iterasi += 1
            all_states_normalized = DDQN.normalize(all_states) # Normalize the state
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            all_mask_actions = MASKING_action(all_states, env)
            if np.random.rand() < DDQN.epsilon:
                if episode % 3 == 0:
                    all_actions = FCFS_action(all_states, env)    
                else:
                    all_actions = RANDOM_action(all_states, env) 
            else:
                all_actions = []
                for r in range(num_agents):
                    all_actions.append(DDQN.select_action_with_masking(r, all_states_normalized[r], all_mask_actions[r]))

            if None in all_actions:
                print("FAILED ACTION: ", all_actions)
                break

            all_next_states, all_rewards, done, truncated, info = env.step(all_actions)
            all_next_states_normalized = DDQN.normalize(all_next_states)

            if env.FAILED_ACTION:
                print("FAILED ENV")
                print("failed RL")
                print("state: ", all_states)
                print("mask_actions: ", all_mask_actions)
                print("joint_actions: ", all_actions)
                print("next_state: ", all_next_states)
                break

            done = done or truncated
            reward_episode += all_rewards
            all_mask_actions_next = MASKING_action(all_next_states, env)
            for r in range(num_agents):
                if not np.array_equal(all_next_states[r], all_states[r]):
                    DDQN.update_RL_memory(r, all_states[r], all_actions[r], all_rewards[r], all_next_states[r], done, all_mask_actions[r], all_mask_actions_next[r])
                if len(DDQN.replay_buffer_dict[r]) >= DDQN.batch_size:
                    DDQN.train_dqn(r)

            all_states = all_next_states
            #--------------------------------------------------------------------------------------------------------

        if env.FAILED_ACTION or None in all_actions:
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
