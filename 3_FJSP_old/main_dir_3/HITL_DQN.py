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

from env_1 import FJSPEnv  # Your flexible job shop environment
from RULED_BASED import MASKING_action, HITL_action

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
        self.save_dir = os.path.join(self.current_dir, 'model_HITL_1')
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
        # Use a deque for a buffer that stores individual (joint) transitions.
        self.replay_buffer = deque(maxlen=buffer_length)
        self.replay_buffer_human = deque(maxlen=buffer_length)
        self.iterasi = 0
        self.save_every_episode = save_every_episode
        self.cumulative_reward_episode = {}
        self.count_ask_human = {}
        self.human_help = False
        self.Quen = deque(maxlen=10)
        self.th=2
        self.reward_max = 100
        self.episode_hitl = 60

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
            replay_filename_human = os.path.join(self.save_dir, f'replay_buffer_human_episode_{episode}.pkl')
            with open(replay_filename_human, 'wb') as f:
                pickle.dump(self.replay_buffer_human, f)
            print(f'Replay buffer saved to {replay_filename}')


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


        count_filename = os.path.join(self.save_dir, 'A_count_ask_human.json')
        if os.path.exists(count_filename):
            # Jika file sudah ada, muat isinya
            with open(count_filename, 'r') as f:
                existing_count= json.load(f)
        else:
            existing_count = {}
        # Updatebaru
        existing_count[episode] = self.count_ask_human[episode]
        with open(count_filename, 'w') as f:
            json.dump(existing_count, f, indent=4)
        print(f'A_count_ask_human saved to {count_filename}')


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


    def human_in_the_loop(self, state, action, reward_satu_episode, env, episode):
        """
        Memeriksa apakah perlu human-in-the-loop, lalu ambil aksi human jika perlu.
        """
        # Jika tidak lagi di intervensi, cek apakah butuh minta bantuan
        self.ask_human(state, action, reward_satu_episode)

        # Jika perlu bantuan manusia
        if self.human_help:
            action_human = HITL_action(state, env)

        else:
            action_human = None

        return action_human
    
    def ask_human(self, state, action, reward_satu_episode):
        self.human_help = False
        """
        Mengecek kondisi untuk memutuskan apakah perlu minta bantuan manusia.
        Berdasarkan selisih Q1-Q2 dan reward episode.
        """
        #print("action: ", action)
        state = tf.reshape(state, (1, 3,self.state_dim))

        Q1_actual = self.dqn_network(state, training=False)
        Q1_actual  = tf.reduce_sum(Q1_actual * 
                                   tf.one_hot(action, self.action_dim))
       

        Q2_actual = self.target_dqn_network(state, training=False)
        Q2_actual = tf.reduce_sum(
                    Q2_actual * tf.one_hot(action, self.action_dim))

        Is = float(Q1_actual - Q2_actual)

        # Jika Q1 - Q2 lebih besar dari max di self.Quen & reward masih di bawah threshold => human_help
        if len(self.Quen) > 0:
            if Is > max(self.Quen) and reward_satu_episode < self.reward_max / self.th:
                self.human_help = True
            else:
                self.human_help = False
        else:
            self.human_help = False

        self.Quen.append(Is)

    def select_action_with_masking(self, state, action_mask_all):

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.dqn_network(state_tensor)  # shape (1, num_agents, action_dim)
    
        q_values = q_values[0]  # shape (num_agents, action_dim)

        action_mask_tensor = tf.convert_to_tensor(action_mask_all, dtype=tf.bool)

        masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))

        actions = tf.argmax(masked_q_values, axis=1)
        return actions.numpy()

    def update_human_memory(self, state, action, reward, next_state, done):
        """Simpan (s, a) dari aksi manusia ke buffer terpisah."""
        self.replay_buffer_human.append((state, action, reward, next_state, done))

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

        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones
    
    def take_human_minibatch(self):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.replay_buffer_human, self.batch_size)
        # Unpack and stack the multi-agent data from each tuple
        mb_states = tf.convert_to_tensor(np.stack([data[0] for data in minibatch]), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.stack([data[1] for data in minibatch]), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.stack([data[2] for data in minibatch]), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.stack([data[3] for data in minibatch]), dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(np.stack([data[4] for data in minibatch]), dtype=tf.float32)

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

            # Do the same for the target network's Q-values on the next states
            mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)
            mb_target_Q_values_next = tf.reduce_sum(
                mb_target_Q_values_next * tf.one_hot(mb_actions_next, self.action_dim),
                axis=2
            )

            # Get Q-values for current states: shape (batch, num_agents, action_dim)
            mb_Q_values = self.dqn_network(mb_states, training=True)
            # Select the Q-values corresponding to the taken actions (one-hot on the last axis)
            mb_Q_values = tf.reduce_sum(
                mb_Q_values * tf.one_hot(mb_actions, self.action_dim),
                axis=2
            )


            # Compute the target Q-values: each shape here is (batch, num_agents)
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * mb_target_Q_values_next
            loss = tf.reduce_mean(tf.square(y - mb_Q_values))

        grads = tape.gradient(loss, self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        del tape

        if len(self.replay_buffer_human) > self.batch_size:
            mb_states_human, mb_actions_human, _, _, _ = self.take_human_minibatch()
            mb_actions_human = tf.cast(mb_actions_human, tf.int32)
            # Menghitung Advantage loss
            with tf.GradientTape(persistent=True) as tape:
                # Q-value Dari state dan action manusia
                Q_values=self.dqn_network(mb_states_human, training=True)
                mb_actions_predicted = tf.argmax(Q_values, axis=2) 
                Q_human = tf.reduce_sum(Q_values * tf.one_hot(mb_actions_human, self.action_dim), axis=2)

                Q_policy =tf.reduce_sum( Q_values * tf.one_hot( mb_actions_predicted, self.action_dim),axis=2)
                # Advantage loss
                advantage_loss= tf.reduce_mean(tf.square(Q_human - Q_policy))
            # Update Critic backprop
            grads_2 = tape.gradient(advantage_loss, self.dqn_network.trainable_variables)
            self.dqn_optimizer.apply_gradients(zip(grads_2, self.dqn_network.trainable_variables))

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
        human_actions=None
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

            if episode < DDQN.episode_hitl:
                mask_actions = MASKING_action(state, env)

                joint_actions= DDQN.select_action_with_masking(state, mask_actions)
                    
                joint_actions= np.array(joint_actions)
                human_actions = DDQN.human_in_the_loop(state, joint_actions, reward_episode,env, episode)
                joint_actions = human_actions if human_actions is not None else joint_actions
            else:
                mask_actions = MASKING_action(state, env)
                joint_actions= DDQN.select_action_with_masking(state, mask_actions)
                human_actions = None

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
                if human_actions is not None:
                    try:
                        DDQN.count_ask_human[episode] += 1
                    except:
                        DDQN.count_ask_human[episode] = 1

                    DDQN.update_human_memory(state, human_actions, reward, next_state, np.repeat(done, num_agents))
                else:
                    try:
                        DDQN.count_ask_human[episode] += 0
                    except:
                        DDQN.count_ask_human[episode] = 0
                    DDQN.update_RL_memory(state, joint_actions, reward, next_state, np.repeat(done, num_agents))

            if len(DDQN.replay_buffer) >= DDQN.batch_size:
                DDQN.train_double_dqn()

            state = next_state
            #--------------------------------------------------------------------------------------------------------

        if env.FAILED_ACTION or None in joint_actions:
            print("FAILED ")
            break

        DDQN.cumulative_reward_episode[episode] = reward_episode
        DDQN.save_replay_buffer_and_rewards(episode)
        DDQN.save_models(episode)
        print("\n")
        env.render()
        print("Episode complete. Total Reward:", reward_episode, 
              "steps:", env.step_count, 
              "product completed:", len(env.conveyor.product_completed))
