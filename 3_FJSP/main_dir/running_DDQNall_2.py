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

from env_5c_4 import FJSPEnv  # Your flexible job shop environment
from MASKING_ACTION_MODEL import masking_action

class DDQN_model:
    def __init__(self,
                 buffer_length=500000,   # maximum number of transitions to store
                 batch_size=64,         # number of transitions per training batch
                 update_delay=2,
                 lr=0.0001,
                 tau=0.005,
                 gamma=0.98,
                 save_every_episode=20):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'saved_all_2')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.state_dim = 14
        self.action_dim = 5
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.epsilon = 0.9
        self.epsilon_decay = 0.92
        self.update_delay = update_delay
        self.batch_size = batch_size
        # Use a deque for a buffer that stores individual (joint) transitions.
        self.replay_buffer = deque(maxlen=buffer_length)
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}
        self.num_agents = 3

        # Create Dense networks (shared among agents).
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        self.update_target_weights()
        #self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

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
        x = layers.TimeDistributed(layers.Dense(64))(state_input)
        x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
        
        x = layers.TimeDistributed(layers.Dense(64))(x)
        x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
        
        x = layers.TimeDistributed(layers.Dense(64))(x)
        x = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01))(x)
        
        output = layers.TimeDistributed(layers.Dense(self.action_dim, activation='linear'))(x)
        
        model = models.Model(inputs=state_input, outputs=output)
        return model


    def update_target_weights(self):
        # Soft update: target = tau * main + (1-tau) * target.
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)

    def select_action_with_masking(self, state, action_mask_all):
        # For 3 agents, 'state' is an array of shape (num_agents, state_dim)
        # and 'action_mask_all' is an array of shape (num_agents, action_dim)
        if np.random.rand() < self.epsilon:
            dummy_actions = []
            # Iterate over each agent's action mask
            for action_mask in action_mask_all:
                true_indices = np.where(action_mask)[0]
                random_action = random.choice(true_indices)
                dummy_actions.append(random_action)
            #print("dummy_actions: ", dummy_actions)
            return dummy_actions
        else:
            # Add batch dimension: state becomes (1, num_agents, state_dim)
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.dqn_network(state_tensor)  # shape (1, num_agents, action_dim)
            #print("q_values bef: ", q_values)
            q_values = q_values[0]  # shape (num_agents, action_dim)
            #print("q_values: ", q_values)
            action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)
            #print("action_mask_tensor: ", action_mask_tensor)
            masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
            #print("masked_q_values: ", masked_q_values)
            # Compute best action for each agent along the action dimension
            actions = tf.argmax(masked_q_values, axis=1)
            #print("actions: ", actions)
            return actions.numpy()

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update_RL_memory(self, state, action, reward, next_state, done):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def take_RL_minibatch(self):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        # Unpack and stack the multi-agent data from each tuple
        mb_states = tf.convert_to_tensor(np.stack([data[0] for data in minibatch]), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.stack([data[1] for data in minibatch]), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.stack([data[2] for data in minibatch]), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.stack([data[3] for data in minibatch]), dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(np.stack([data[4] for data in minibatch]), dtype=tf.float32)
        # print("mb_states: ", mb_states.shape)
        # print("mb_actions: ", mb_actions.shape)
        # print("mb_rewards: ", mb_rewards.shape)
        # print("mb_next_states: ", mb_next_states.shape)
        # print("mb_dones: ", mb_dones.shape)
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones


    def train_double_dqn(self):
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.take_RL_minibatch()
        # Now, mb_states and mb_next_states are expected to have shape (batch, num_agents, state_dim)
        # and mb_actions, mb_rewards, mb_dones are (batch, num_agents)
        mb_actions = tf.cast(mb_actions, tf.int32)
        with tf.GradientTape() as tape:
            # Get Q-values for the next states: shape (batch, num_agents, action_dim)
            mb_Q_values_next = self.dqn_network(mb_next_states, training=False)
            # For each agent, pick the action with the highest Q-value along the last axis
            mb_actions_next = tf.argmax(mb_Q_values_next, axis=2)  # shape (batch, num_agents)

            # Get Q-values for current states: shape (batch, num_agents, action_dim)
            mb_Q_values = self.dqn_network(mb_states, training=True)
            # Select the Q-values corresponding to the taken actions (one-hot on the last axis)
            mb_Q_values = tf.reduce_sum(
                mb_Q_values * tf.one_hot(mb_actions, self.action_dim),
                axis=2
            )

            # Do the same for the target network's Q-values on the next states
            mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)
            mb_target_Q_values_next = tf.reduce_sum(
                mb_target_Q_values_next * tf.one_hot(mb_actions_next, self.action_dim),
                axis=2
            )

            # Compute the target Q-values: each shape here is (batch, num_agents)
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * mb_target_Q_values_next
            loss = tf.reduce_mean(tf.square(y - mb_Q_values))

        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        del tape

        # Periodically update target network (soft update)
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights()


if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000)
    DDQN = DDQN_model()
    num_agents = env.num_agents
    num_episodes = 1000

    for episode in tqdm(range(1, num_episodes + 1)):
        state, info = env.reset(seed=episode)
        reward_episode = 0
        done = False
        truncated = False
        print("\nEpisode:", episode)

        # For joint transitions, we initialize a single episode buffer.
        
        while not done:
            DDQN.iterasi += 1
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            mask_actions = masking_action(state, env)

            joint_actions = DDQN.select_action_with_masking(state, mask_actions)
                
            joint_actions = np.array(joint_actions)
            if None in joint_actions:
                print("FAILED ACTION: ", joint_actions)
                break

            next_state, reward, done, truncated, info = env.step(joint_actions)

            if env.FAILED_ACTION:
                print("FAILED ENV")
                break

            done = done or truncated
            reward_episode += np.mean(reward) 

            if not np.array_equal(next_state, state):
                DDQN.update_RL_memory(state, joint_actions, reward, next_state, np.repeat(done, num_agents))

            if len(DDQN.replay_buffer) >= DDQN.batch_size:
                for _ in range(2):
                    DDQN.train_double_dqn()

            state = next_state
            #--------------------------------------------------------------------------------------------------------

        if env.FAILED_ACTION or None in joint_actions:
            print("FAILED ")
            break

        DDQN.update_epsilon()
        DDQN.cumulative_reward_episode[episode] = reward_episode
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed))
