from env_5c_3 import FJSPEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import os
import pickle
import json
from collections import deque
from tqdm import tqdm

class DDQN_model:
    def __init__(self,
        buffer_length=500000,
        batch_size=4,
        update_delay=2,
        lr=0.0001,
        tau=0.005,
        gamma=0.98,
        ):
        self.state_dim= 14
        self.action_dim=5
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.epsilon=1
        self.epsilon_decay=0.92
        self.update_delay=update_delay
        self.batch_size=batch_size
    
        self.dqn_network= self.create_dqn_network()
        self.target_dqn_network =self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        self.update_target_weights()
        # Optimizer
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.memory_B = deque(maxlen=int(buffer_length))

    def save_models(self, episode):
        """
        Menyimpan model.
        """
        dqn_save_path = os.path.join(self.save_dir, f'dqn_episode_{episode}.h5')
        target_dqn_save_path = os.path.join(self.save_dir, f'target_dqn_episode_{episode}.h5')
        self.dqn_network.save_weights(dqn_save_path)
        self.target_dqn_network.save_weights(target_dqn_save_path)
        print(f'Models saved at episode {episode}')


    def save_replay_buffer_and_rewards(self, episode):
        """
        Menyimpan replay buffer ke file pickle dan reward kumulatif ke file JSON.
        """
        # Simpan replay buffer setiap beberapa episode
        if episode % self.save_every_episode == 0:
            replay_filename = os.path.join(self.save_dir, f'replay_buffer_episode_{episode}.pkl')
            with open(replay_filename, 'wb') as f:
                pickle.dump(self.memory_B, f)
            print(f'Replay buffer saved to {replay_filename}')

        # Simpan reward kumulatif setiap episode
        rewards_filename = os.path.join(self.save_dir, 'A_cumulative_rewards.json')
        if os.path.exists(rewards_filename):
            # Jika file sudah ada, muat isinya
            with open(rewards_filename, 'r') as f:
                existing_rewards = json.load(f)
        else:
            existing_rewards = {}

        # Update dengan reward baru
        existing_rewards[episode] = self.cumulative_reward_episode[episode]
        with open(rewards_filename, 'w') as f:
            json.dump(existing_rewards, f, indent=4)
        print(f'Cumulative rewards saved to {rewards_filename}')

    def create_dqn_network(self):
        """Membuat model Q-network dengan layer fully connected."""
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model
    
    def take_RL_minibatch(self):
        """
        Sample a minibatch from the replay buffer.
        Each minibatch element contains multi-agent data:
          - state: (3, 14)
          - action: (3,)
          - reward: (3,)
          - next_state: (3, 14)
          - done: (3,)
        After stacking, the shapes become:
          - mb_states: (batch_size, 3, 14)
          - mb_actions: (batch_size, 3)
          - mb_rewards: (batch_size, 3)
          - mb_next_states: (batch_size, 3, 14)
          - mb_dones: (batch_size, 3)
        """
        minibatch = random.sample(self.memory_B, self.batch_size)
        
        # Unpack and stack the multi-agent data from each tuple
        mb_states = tf.convert_to_tensor(np.stack([data[0] for data in minibatch]), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.stack([data[1] for data in minibatch]), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.stack([data[2] for data in minibatch]), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.stack([data[3] for data in minibatch]), dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(np.stack([data[4] for data in minibatch]), dtype=tf.float32)
        # Reshape the multi-agent data to flatten the batch dimension
        mb_states = tf.reshape(mb_states, (-1, self.state_dim))
        mb_actions = tf.reshape(mb_actions, (-1,))
        mb_rewards = tf.reshape(mb_rewards, (-1,))
        mb_next_states = tf.reshape(mb_next_states, (-1, self.state_dim))
        mb_dones = tf.reshape(mb_dones, (-1,))

        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones
    
    def update_RL_memory(self, state, action, reward, next_state, done):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.memory_B.append((state, action, reward, next_state, done))
    
    def update_target_weights(self):
        """
        Soft update pada target network.
        """
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)
            

    def select_action_with_masking(self, state, action_mask):
        if np.random.rand() < self.epsilon:
            #tf.print("Random action")
            action_mask_index = np.where(action_mask)[0]
            return random.choice(action_mask_index)
        else:
            #tf.print("NN action")
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.dqn_network(state_tensor)
            action_mask_tensor = tf.convert_to_tensor(action_mask)
            masked_q_values = tf.where(action_mask_tensor, q_values, tf.fill(tf.shape(q_values), -np.inf))
            ##tf.print("masked_q_values: ", masked_q_values)
            ##choose_action = tf.argmax(masked_q_values, axis=1)
            ##tf.print("choose_action: ", choose_action)
            return int(tf.argmax(masked_q_values, axis=1))
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def train_double_dqn(self):
        #tf.print("\n")
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.take_RL_minibatch()
        #tf.print("mb_actions_before: ", mb_actions)
        mb_actions = tf.cast(mb_actions, tf.int32)
        # tf.print("mb_actions: ", mb_actions)
        # tf.print("tf.one_hot(mb_actions, self.action_dim): ", tf.one_hot(mb_actions, self.action_dim))
        with tf.GradientTape() as tape:
            mb_Q_values_next= self.dqn_network(mb_next_states, training=False)
            mb_actions_next = tf.argmax(mb_Q_values_next, axis=1)

            mb_Q_values= self.dqn_network(mb_states, training=True)
            mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), 
                                        axis=1)


            #mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)[mb_actions_next]
        
            mb_target_Q_values_next = self.target_dqn_network(mb_next_states, training=False)
            mb_target_Q_values_next = tf.reduce_sum(mb_target_Q_values_next * 
                                                    tf.one_hot(mb_actions_next, self.action_dim), 
                                                    axis=1)
            
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * mb_target_Q_values_next
            loss = tf.reduce_mean(tf.square(y - mb_Q_values))

        # Menghitung gradien dan mengupdate bobot model
        grads = tape.gradient(loss,  self.dqn_network.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(grads, self.dqn_network.trainable_variables))
        del tape
        
        # Periodically update target network (soft update)
        if self.iterasi % self.update_delay == 0:
            self.update_target_weights()



def masking_action(states, env):
    mask_actions=[]

    for i, state in enumerate(states):
        is_agent_working=False
        is_status_idle=False
        is_status_accept=False
        is_pick_job_window_yr_1=False
        is_pick_job_window_yr_2=False
        is_job_in_capability_yr=False
        is_job_in_capability_yr_1=False
        is_job_in_capability_yr_2=False
        #[Accept, Wait yr-1, Wait yr-2, Decline, Continue]
        accept_action=False
        wait_yr_1_action=False
        wait_yr_2_action=False
        decline_action=False
        continue_action=False
        if state[env.state_operation_now_location] !=0:
            is_agent_working=True
        if state[env.state_status_location_all[i]] ==0:
            is_status_idle=True
        if state[env.state_status_location_all[i]] ==1:
            is_status_accept=True
        if state[env.state_pick_job_window_location] ==1:
            is_pick_job_window_yr_1=True
        if state[env.state_pick_job_window_location]==2:
            is_pick_job_window_yr_2=True
        if (state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]):
            is_job_in_capability_yr=True
        if (state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location]):
            is_job_in_capability_yr_1=True
        if (state[env.state_first_job_operation_location[2]]in state[env.state_operation_capability_location]):
            is_job_in_capability_yr_2=True

        #[Accept, Wait yr-1, Wait yr-2, Decline, Continue]
        if is_status_accept:
            accept_action=True

        elif not is_agent_working:
            if is_status_idle and is_pick_job_window_yr_1 and is_job_in_capability_yr:
                accept_action=True
                decline_action=True
                continue_action=True

            if is_status_idle and is_job_in_capability_yr_2:
                wait_yr_2_action=True
                decline_action=True

            if is_status_idle and is_job_in_capability_yr_1:
                wait_yr_1_action=True
                decline_action=True
            
            # if not is_job_in_capability_yr and not is_job_in_capability_yr_1 and not is_job_in_capability_yr_2:
            #     continue_action=True
            continue_action=True


        elif is_agent_working:
            if not is_status_idle:
                continue_action=True

        mask_actions.append([accept_action, wait_yr_1_action, wait_yr_2_action, decline_action, continue_action])
    return mask_actions

        

       

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=10)
    DDQN=DDQN_model()
    for episode in tqdm(range(1, 1+ 1)):
        state, info = env.reset(seed=episode)
        reward_satu_episode = 0
        done = False
        truncated = False
        print("\nEpisode:", episode)
        
        if env.FAILED_ACTION:
            print("FAILED ENV")
            break

        while not done:
            
            if len(env.conveyor.product_completed)>= env.n_jobs:
                print("All jobs are completed.")
                break
            if env.step_count >= env.max_steps:
                print("Max steps reached.")
                break

            mask_actions=masking_action(state, env)
            actions=[]
            # Multi-agent shared Neural Network
            for single, mask_action in zip(state, mask_actions):
                action = DDQN.select_action_with_masking(single, mask_action)
                actions.append(action)
                #dummy +=1
            actions = np.array(actions)

            if None in actions:
                print("FAILED ACTION: ", actions)
                break

            next_state, reward, done, truncated, info = env.step(actions)

            if env.FAILED_ACTION:
                print("FAILED ENV")
                break
            done_buffer = np.array([done or truncated]*env.num_agents)

            if  np.array_equal(state, next_state):
                print("State is not equal")
                DDQN.update_RL_memory(state, actions, reward, next_state, done_buffer)
            if len(DDQN.memory_B) >= DDQN.batch_size:

                mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = DDQN.take_RL_minibatch()
                # print("States batch shape:", mb_states.shape)
                # print("Actions batch shape:", mb_actions.shape)
                # print("Rewards batch shape:", mb_rewards.shape)
                # print("Next States batch shape:", mb_next_states.shape)
                # print("Dones batch shape:", mb_dones.shape)
                print("\n")
            #DDQN.train_double_dqn()
            reward_satu_episode += reward
            state = next_state

        #-------------------------------------------------
        print("Episode complete. Total Reward:", reward_satu_episode, 
              "jumlah step:", env.step_count, 
              "product completed: ",len(env.conveyor.product_completed))
        order = {'A': 0, 'B': 1, 'C': 2}

        print("product completed: ",env.conveyor.product_completed)
        sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

        #print("product sorted: ",sorted_jobs)