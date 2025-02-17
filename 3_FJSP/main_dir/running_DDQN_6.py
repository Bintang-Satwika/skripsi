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
        self.save_dir = os.path.join(self.current_dir, 'saved_xx')
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
        # Dense network: input shape is (state_dim,)
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation="relu")(state_input)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def update_target_weights(self):
        # Soft update: target = tau * main + (1-tau) * target.
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)

    def select_action_with_masking(self, state, action_mask):
        # 'state' is a vector of shape (state_dim,)
        if np.random.rand() < self.epsilon:
            valid_actions = np.where(action_mask)[0]
            return int(random.choice(valid_actions))
        else:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)  # shape: (1, state_dim)
            q_values = self.dqn_network(state_tensor)  # shape: (1, action_dim)
            action_mask_tensor = tf.convert_to_tensor(action_mask, dtype=tf.bool)
            masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
            return int(tf.argmax(masked_q_values, axis=1)[0])
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def take_minibatch(self):
        # Sample a minibatch of joint transitions from the replay buffer.
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(list(self.replay_buffer), self.batch_size)
        # Each element is a tuple: (joint_state, joint_action, joint_reward, joint_next_state, done)
        joint_states, joint_actions, joint_rewards, joint_next_states, dones = zip(*batch)
        # Convert to arrays:
        joint_states = np.array(joint_states)         # shape: (batch, num_agents, state_dim)
        print(joint_states.shape)
        joint_actions = np.array(joint_actions)         # shape: (batch, num_agents)
        joint_rewards = np.array(joint_rewards)         # shape: (batch, num_agents)
        joint_next_states = np.array(joint_next_states) # shape: (batch, num_agents, state_dim)
        dones = np.array(dones).astype(np.float32)      # shape: (batch,)
        return joint_states, joint_actions, joint_rewards, joint_next_states, dones

    def train_double_dqn(self):
        minibatch = self.take_minibatch()
        if minibatch is None:
            return  # not enough data
        
        joint_states, joint_actions, joint_rewards, joint_next_states, dones = minibatch
        # Assume there are num_agents transitions in each sample.
        batch_size, num_agents, _ = joint_states.shape

        # Flatten the batch so that each agent's sample is treated independently.
        flat_states = joint_states.reshape(-1, self.state_dim)           # shape: (batch*num_agents, state_dim)
        flat_actions = joint_actions.reshape(-1)                           # shape: (batch*num_agents,)
        flat_rewards = joint_rewards.reshape(-1)                           # shape: (batch*num_agents,)
        flat_next_states = joint_next_states.reshape(-1, self.state_dim)     # shape: (batch*num_agents, state_dim)
        # Expand dones for each agent:
        flat_dones = np.repeat(dones, num_agents)                          # shape: (batch*num_agents,)

        with tf.GradientTape() as tape:
            q_values = self.dqn_network(flat_states, training=True)  # shape: (batch*num_agents, action_dim)
            q_values_taken = tf.reduce_sum(q_values * tf.one_hot(flat_actions, self.action_dim), axis=1)
            # Double DQN: use main network to select next action:
            next_q_main = self.dqn_network(flat_next_states, training=False)
            next_actions = tf.argmax(next_q_main, axis=1)
            next_q_target = self.target_dqn_network(flat_next_states, training=False)
            next_q_target_taken = tf.reduce_sum(next_q_target * tf.one_hot(next_actions, self.action_dim), axis=1)
            # Compute targets:
            y = flat_rewards + (1.0 - flat_dones) * self.gamma * next_q_target_taken
            loss = tf.reduce_mean(tf.square(y - q_values_taken))
        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
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
        episode_buffer = []  # each element: (joint_state, joint_action, joint_reward, joint_next_state, done)
        
        while not done:
            DDQN.iterasi += 1
            if len(env.conveyor.product_completed) >= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            mask_actions = masking_action(state, env)
            joint_actions = []
            for single_state, mask in zip(state, mask_actions):
                action = DDQN.select_action_with_masking(single_state, mask)
                joint_actions.append(action)
                
            joint_actions = np.array(joint_actions)
            if None in joint_actions:
                print("FAILED ACTION: ", joint_actions)
                break

            next_state, reward, done, truncated, info = env.step(joint_actions)
            done = done or truncated
            reward_episode += np.mean(reward) 
            if env.FAILED_ACTION:
                print("FAILED ENV")
                break

            # Only store transition if next_state is different from state.
            if not np.array_equal(next_state, state):
                # Each transition is stored as a joint tuple.
                transition = (state, joint_actions, reward, next_state, done)
                episode_buffer.append(transition)
            state = next_state
            for _ in range(2):
                DDQN.train_double_dqn()

        if env.FAILED_ACTION:
            print("FAILED ENV, skipping episode storage.")
            break

        # After episode ends, add the episode_buffer to the replay buffer.
        for transition in episode_buffer:
            DDQN.replay_buffer.append(transition)
        DDQN.update_epsilon()
        DDQN.cumulative_reward_episode[episode] = reward_episode
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed))
