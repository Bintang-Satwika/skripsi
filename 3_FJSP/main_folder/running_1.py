from env_5c import FJSPEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import os
import pickle
import json

class DDQN_model:
    def __init__(self):
        self.dqn_network= self.create_dqn_network()
        self.target_dqn_network =self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        # Optimizer
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.epsilon=1
        self.epsilon_decay=0.9

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
            

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.dqn_network(state_tensor)
            return int(tf.argmax(q_values[0]).numpy())
    
    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

def FCFS_action(states):
    actions=[]
    for i, state in enumerate(states):
        if state[env.state_operation_now_location]==0:

            if state[env.state_status_location_all[i]]==1:
                actions.append(0) # accept saat di workbench
            elif (state[env.state_first_job_operation_location[0]]!=0 and 
                  state[env.state_first_job_operation_location[0]] in state[env.state_operation_capability_location]
                  and state[env.state_pick_job_window_location] ==1 ):
                actions.append(0) # Accept saat di conveyor yr
            elif (state[env.state_first_job_operation_location[1]]!=0 and 
                  state[env.state_first_job_operation_location[1]] in state[env.state_operation_capability_location]):# ada job di yr-1
                actions.append(1) # wait yr-1
            elif (state[env.state_first_job_operation_location[2]]!=0 and 
                  state[env.state_first_job_operation_location[2]] in state[env.state_operation_capability_location]):# ada job di yr-2
                actions.append(2) # wait yr-2
            elif np.array_equal(state[env.state_first_job_operation_location], [0, 0, 0]):
                actions.append(4) # continue
            else:
                print("milih decline")
                actions.append(3) # Decline

        elif state[env.state_operation_now_location]!=0:
            if state[env.state_status_location_all[i]]==2 or  state[env.state_status_location_all[i]]==3 : # agent working hingga completing
                actions.append(4) # continue
            else:
                print("FAILED ACTION: agent status is not working")
                actions.append(None)

        else:
            actions.append(None)
            print("PASTI ADA YANG SALAH")
    return actions


def masking_action(states, env):
    mask_actions=[]

    for i, state in enumerate(states):
        print("agent-", i+1)
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
        print
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
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=500)
    state, info = env.reset(seed=3)
    #nv.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial state:", state)
    while not done and not truncated:
        if len(env.conveyor.product_completed)>= env.n_jobs:
            print("All jobs are completed.")
            break
        if env.FAILED_ACTION:
            print("FAILED ENV")
            break
        print("\nStep:", env.step_count)
        # Untuk contoh, gunakan aksi acak
        #actions = env.action_space.sample()

        mask_actions=masking_action(state, env)
        actions=[]
        for state, mask_action in zip(state, mask_actions):
            print("state: ", state)
            print("mask action: ", mask_action)
            true_indices = np.where(mask_action)[0]
            random_actions = random.choice(true_indices)
            print("random actions: ", random_actions)
            
            actions.append(random_actions)


        if None in actions:
            print("FAILED ACTION: ", actions)
            break
        #print("state: ", state)
        print("Actions:", actions)
        next_state, reward, done, truncated, info = env.step(actions)
        #print("Reward:", reward)
        print("NEXT STATE:", next_state)
        total_reward += reward
        env.render()
        print()
        print("-" * 100)
        state = next_state
    print("len(env.conveyor.product_completed)", len(env.conveyor.product_completed))
    print("Episode complete. Total Reward:", total_reward, "jumlah step:", env.step_count)
    order = {'A': 0, 'B': 1, 'C': 2}

    # Sorting by product type first, then by numeric value
    sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

    print("product sorted: ",sorted_jobs)