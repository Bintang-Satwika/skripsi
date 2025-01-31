import gymnasium as gym
from tqdm import tqdm
import random
import time
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras import layers, models
from collections import deque
from matplotlib import pyplot as plt
import json
import highway_env
from highway_env.vehicle.kinematics import Vehicle
Vehicle.MIN_SPEED = -5  # Atur kecepatan minimum menjadi -5 m/s
Vehicle.MAX_SPEED = 30  # Atur kecepatan maksimum menjadi 20 m/s
print(Vehicle.on_road)

from pynput import keyboard

tf.keras.backend.set_floatx('float32')

class TD3Agent:
    def __init__(
        self,
        env_name="LunarLander-v3",
        n_episodes=400,
        save_every_episode=20,
        buffer_length=100000,
        human_buffer_length=None,
        batch_size=256,
        update_delay=2,
        lr=0.0001,
        tau=0.005,
        gamma=0.99,
        th=2,
        n=15,
        sigma_init=0.4,
        sigma_target=0.2,
        noise_clip=0.5,
        seed=10,
        render_mode='human'
    ):
        """
        Inisialisasi agent TD3 dan hyperparameter.
        """

        # Set seed 
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Direktori untuk menyimpan model dan buffer
        self.save_dir = 'double_dqn/highwaymodel_2'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Environment
        columns_selected = ["presence", "x", "y", "vx", "vy"]
        config = {
            "vehicles_count": 10,
            "observation": {
                
                "type": "Kinematics",
                
                "vehicles_count": 10,
                "features": columns_selected,
                "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-5, 30],
                        "vy": [-5, 30]
                    },
                "absolute": False,
                "order": "sorted",
                "normalize": True,
                "clip": True,
                "include_obstacle": True,
                "observe_intentions": True,
            },
            "action": {
                "type": "DiscreteMetaAction"
                #"type": "ContinuousAction"
            },
            "lanes_count": 4,
            'show_trajectories': False,
            "manual_control": False,
            "reward_speed_range": [20, 30],
            "collision_reward": -100,
            "duration": 300,
             "normalize_reward": False
        }
        self.env = gym.make('highway-fast-v0', render_mode='rgb_array', config=config)
        self.n_episodes = n_episodes
        self.last_episode= 0

        #  untuk merekam statistik episode
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=n_episodes)

        # Dimensi state dan action
        self.state_dim, info = self.env.reset()
        self.state_dim=np.ravel(self.state_dim)
        self.state_dim = self.state_dim.shape[0]
        self.action_dim = self.env.action_space.n

        print(f"Dimensi State: {self.state_dim}, Dimensi Aksi: {self.action_dim}")

        # Hyperparameter TD3
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.sigma = sigma_init
        self.sigma_target = sigma_target
        self.noise_clip = noise_clip
        self.update_delay = update_delay
        self.iterasi = 0
        self.batch_size = batch_size
        self.th = th
        self.n = n
        self.save_every_episode =  save_every_episode
        self.cumulative_reward_episode = {}
        self.human_help = False

        # Replay buffer
        self.memory_B = deque(maxlen=int(buffer_length))

        
        self.dqn_network= self.create_dqn_network()
        self.target_dqn_network =self.create_dqn_network()
        self.target_dqn_network.set_weights(self.dqn_network.get_weights())
        # Optimizer
        self.dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)



        # --------------------------------------
        # Variabel untuk menangani keyboard
        # --------------------------------------
        self.pressed_keys = set()  # set penampung tombol yg sedang ditekan
        self.stop_training = False  # jika True => hentikan loop
        self.listener = None  # akan diisi instance keyboard.Listener


    # =============================
    # Bagian Keyboard dengan pynput
    # =============================
    def on_press(self, key):
        """
        Callback ketika tombol keyboard ditekan.
        Kita tambahkan ke set 'pressed_keys' jika relevan.
        """
        if key == keyboard.Key.esc:
            self.stop_training = True

    def on_release(self, key):
        """
        Callback ketika tombol keyboard dilepas.
        Buang tombol tersebut dari set 'pressed_keys'.
        """
        try:
            if key.char in self.pressed_keys:
                self.pressed_keys.remove(key.char)
        except:
            pass

    def start_keyboard_listener(self):
        """
        Memulai listener keyboard (asinkron).
        """
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def stop_keyboard_listener(self):
        """
        Menghentikan listener keyboard.
        """
        if self.listener is not None:
            self.listener.stop()
            self.listener = None


    # ===================
    # Replay Memory Utils
    # ===================
    def update_RL_memory(self, state, action, reward, next_state, done):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.memory_B.append((state, action, reward, next_state, done))
    

    def take_RL_minibatch(self):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.memory_B, self.batch_size)
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = zip(*minibatch)
        mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(mb_dones, dtype=tf.float32)
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones

    def update_target_weights(self):
        """
        Soft update pada target network.
        """
        for main_var, target_var in zip(self.dqn_network.trainable_variables,
                                        self.target_dqn_network.trainable_variables):
            target_var.assign(self.tau * main_var + (1.0 - self.tau) * target_var)



    # =============
    # Save / Logging
    # =============
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
        """
        Membuat model.
        """
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model
    
    def train_double_dqn(self):
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.take_RL_minibatch()
        mb_actions = tf.cast(mb_actions, tf.int32)

        with tf.GradientTape() as tape:
            mb_Q_values= self.dqn_network(mb_states, training=True)
            mb_Q_values = tf.reduce_sum(mb_Q_values * tf.one_hot(mb_actions, self.action_dim), 
                                        axis=1)

            mb_Q_values_next= self.dqn_network(mb_next_states, training=False)
            mb_actions_next = tf.argmax(mb_Q_values_next, axis=1)

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
   
    def select_action(self, state, epsilon=0.3):
        if np.random.rand() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.dqn_network(state_tensor)
            return int(tf.argmax(q_values[0]).numpy())

    # =======================
    # Training / Main Loop
    # =======================

    def main_loop(self):
        """
        Loop utama untuk melatih agent.
        """
        # Mulai listener keyboard
        self.start_keyboard_listener()

        for episode in tqdm(range(1, self.n_episodes + 1)):
            if self.stop_training:
                # Jika user menekan ESC sebelum mulai episode
                break

            state, info = self.env.reset()
            state=np.ravel(state)
            done = False
            reward_satu_episode = 0
            action_RL = None
            self.human_step_now = 0
            self.step_no_keyboard_now = 0
            max_iterasi_episode= 0

            while not done:

                if max_iterasi_episode >= 1000:
                    print("Episode: ", episode, "Terlalu lama")
                    break
                if self.stop_training:
                    # Jika user menekan ESC di tengah episode
                    done = True
                    break

                max_iterasi_episode += 1
                self.iterasi += 1

                action_RL = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action_RL)
                next_state=np.ravel(next_state)
                reward_satu_episode += reward
                done = terminated or truncated

                # Simpan ke buffer
                self.update_RL_memory(state, action_RL, reward, next_state, done)
                # Update state
                state = next_state
                # Train model
                if len(self.memory_B) > self.batch_size:
                    self.train_double_dqn()
                # Render environment
                self.env.render()

            print(f"\nEpisode: {episode}, Reward: {reward_satu_episode}, Iterasi: {self.iterasi}")
            # Simpan reward kumulatif
            self.cumulative_reward_episode[episode] = reward_satu_episode
            self.save_replay_buffer_and_rewards(episode)
            # Simpan model
            if episode % self.save_every_episode == 0:
                self.save_models(episode)

            if self.stop_training:
                break

        # Hentikan listener keyboard & tutup environment
        self.last_episode = str(episode)+"ESC"
        self.save_models(self.last_episode)
        self.stop_keyboard_listener()
        self.env.close()



    def plot_results(self):
        """
        Plot reward dan panjang episode setelah training selesai.
        """
        episode_rewards = np.array(self.env.return_queue)
        episode_lengths = np.array(self.env.length_queue)

        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # Plot Episode Rewards
        axs[0].plot(episode_rewards)
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        # Plot Episode Lengths
        axs[1].plot(episode_lengths)
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Length")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = TD3Agent()
    agent.main_loop()
    agent.plot_results()
