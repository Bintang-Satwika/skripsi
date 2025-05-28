import os
import random
import pickle
import json
from collections import deque
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from env_1_testing import FJSPEnv  # Your flexible job shop environment
from RULED_BASED import MASKING_action, HITL_action, FCFS_action, RANDOM_action


class DDQN_model:
    def __init__(self,
                 buffer_length=500000,
                 batch_size=128,
                 update_delay=1,
                 lr=0.0001,
                 tau=0.01,
                 gamma=0.98,
                 save_every_episode=50,
                 episode_load=0):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.current_dir, 'DQN_yr2_2')
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_agents = 3
        self.state_dim = 13
        self.action_dim = 5
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_delay = update_delay
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.992
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}
        self.episode_load = episode_load

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_length)

        # Networks
        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())

        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        print('Initialized DDQN_model. Buffer size:', len(self.replay_buffer))

    def load_models(self, load_dir, episode):
        dqn_path = os.path.join(load_dir, f'dqn_episode_{episode}.h5')
        tgt_path = os.path.join(load_dir, f'target_dqn_episode_{episode}.h5')
        if not os.path.exists(dqn_path) or not os.path.exists(tgt_path):
            raise FileNotFoundError('Model file not found')
        self.dqn_network.load_weights(dqn_path)
        self.target_dqn_network.load_weights(tgt_path)
        print(f'Models loaded from episode {episode}')

    def save_models(self, episode):
        if episode % self.save_every_episode == 0:
            self.dqn_network.save_weights(os.path.join(self.save_dir, f'dqn_episode_{episode}.h5'))
            self.target_dqn_network.save_weights(os.path.join(self.save_dir, f'target_dqn_episode_{episode}.h5'))
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
        state_input = layers.Input(shape=(self.num_agents, self.state_dim))
        x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(state_input)
        x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
        output = layers.TimeDistributed(layers.Dense(self.action_dim, activation='linear'))(x)
        return models.Model(inputs=state_input, outputs=output)

    def update_target_weights(self):
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)

    @tf.function
    def select_action_with_masking(self, state, action_mask_all):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.dqn_network(state_tensor)
        q_values = q_values[0]
        mask = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)
        neg_inf = tf.fill(tf.shape(q_values), -np.inf)
        masked_q = tf.where(mask, q_values, neg_inf)
        return tf.argmax(masked_q, axis=1)

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def update_RL_memory(self, state, action, reward, next_state, done, mask_actions, mask_actions_next):
        self.replay_buffer.append((state, action, reward, next_state, done, mask_actions, mask_actions_next))

    def take_RL_minibatch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = tf.convert_to_tensor(np.stack([b[0] for b in batch]), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.stack([b[1] for b in batch]), dtype=tf.int32)
        rewards = tf.convert_to_tensor(np.stack([b[2] for b in batch]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.stack([b[3] for b in batch]), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.stack([b[4] for b in batch]), dtype=tf.float32)
        masks_next = tf.convert_to_tensor(np.stack([b[6] for b in batch]), dtype=tf.bool)
        return states, actions, rewards, next_states, dones, masks_next

    @tf.function
    def _train_step(self, mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones, mb_action_mask_next):
        with tf.GradientTape() as tape:
            # Q(s,a)
            q_vals = self.dqn_network(mb_states, training=True)
            q_taken = tf.reduce_sum(q_vals * tf.one_hot(mb_actions, self.action_dim), axis=2)
            # Q'(s',a')
            q_next = self.target_dqn_network(mb_next_states, training=False)
            neg_inf = tf.fill(tf.shape(q_next), -np.inf)
            q_next_masked = tf.where(mb_action_mask_next, q_next, neg_inf)
            max_q_next = tf.reduce_max(q_next_masked, axis=2)
            # Bellman target
            target = mb_rewards + (1.0 - mb_dones) * self.gamma * max_q_next
            loss = tf.reduce_mean(tf.square(target - q_taken))
        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        return loss

    def train_dqn(self):
        states, actions, rewards, next_states, dones, mask_next = self.take_RL_minibatch()
        loss = self._train_step(states, actions, rewards, next_states, dones, mask_next)
        self.iterasi += 1
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights()
        return loss

    def normalize(self, state):
        mins = np.array([3,1,1,0,0,0,0,0,0,0,0,0,0,])
        maxs = np.array([11,3,3,3,3,3,3,3,3,3,3,3,3,])
        return np.float32((state - mins) / (maxs - mins))


if __name__ == "__main__":
    DDQN = DDQN_model()
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=1000, episode=DDQN.episode_load)
    for ep in tqdm(range(DDQN.episode_load+1, 801)):
        state, _ = env.reset(seed=ep)
        state = state[:, :-1]
        if (ep-1) %1 == 0:
            episode_seed= ep-1

        env.conveyor.episode_seed= episode_seed
        print(env.conveyor.episode_seed)
        reward_satu_episode = 0
        done = False
        truncated = False
        total_reward = 0.0
        while not done: 
            st = DDQN.normalize(state)
            mask = MASKING_action(state, env)
            if random.random() < DDQN.epsilon:
                actions = RANDOM_action(state, env)   # Episode 3, 6, 9, ...
            else:
                actions = DDQN.select_action_with_masking(st, mask)
                actions = actions.numpy()
            next_state, reward, done, truncated, info = env.step(actions)
            next_state = next_state[:, :-1]
            done = done or truncated
            
            total_reward += np.mean(reward)
            next_norm = DDQN.normalize(next_state)
            mask_next = MASKING_action(next_state, env)
            DDQN.update_RL_memory(st, actions, np.repeat(np.mean(reward), DDQN.num_agents), next_norm,
                                  np.repeat(done or truncated, DDQN.num_agents), mask, mask_next)
            if len(DDQN.replay_buffer) >= DDQN.batch_size:
                DDQN.train_dqn()
            state = next_state
        DDQN.update_epsilon()
        DDQN.cumulative_reward_episode[ep] = total_reward
        DDQN.save_replay_buffer_and_rewards(ep)
        DDQN.save_models(ep)
        print(f"Episode {ep} done, reward={total_reward}, epsilon={DDQN.epsilon}")
