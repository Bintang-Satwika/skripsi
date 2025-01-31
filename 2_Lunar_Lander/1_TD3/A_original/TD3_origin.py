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


        # Direktori untuk menyimpan model dan buffer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = 'dummy'
        self.save_dir = os.path.join(current_dir, self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Environment
        self.env = gym.make(
            env_name, 
            continuous=True,
            gravity=-10,
            enable_wind=True,
            wind_power=15,
            turbulence_power=1.5,
            render_mode=render_mode
        )
        self.n_episodes = n_episodes
        self.last_episode= 0
        # self.imitation_episodes = 5
        # self.hitl_episodes = 50

        #  untuk merekam statistik episode
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=n_episodes)

        # Dimensi state dan action
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

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

        # Inisialisasi jaringan (Actor dan Critic)
        self.actor = self.create_actor_network()
        self.critic_1 = self.create_critic_network()
        self.critic_2 = self.create_critic_network()

        # Optimizer
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Target network
        self.target_actor = self.create_actor_network()
        self.target_actor.set_weights(self.actor.get_weights())

        self.target_critic_1 = self.create_critic_network()
        self.target_critic_1.set_weights(self.critic_1.get_weights())

        self.target_critic_2 = self.create_critic_network()
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        # --------------------------------------
        # Variabel untuk menangani keyboard
        # --------------------------------------
        self.pressed_keys = set()  # set penampung tombol yg sedang ditekan
        self.stop_training = False  # jika True => hentikan loop
        self.listener = None  # akan diisi instance keyboard.Listener


    # ================================
    # Neural Network Actor & Critic
    # ================================
    def create_actor_network(self):
        """
        Membuat model Actor dengan output di [-1, 1] untuk setiap dimensi aksi.
        """
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='tanh')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def create_critic_network(self):
        """
        Membuat model Critic (Q-network). 
        Input: state dan action, output: Q-value.
        """
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(1)(x)
        model = models.Model(inputs=[state_input, action_input], outputs=output)
        return model

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

    # =================================================
    # Fungsi seleksi action RL dan target policy
    # =================================================
    def select_action(self, state):
        """
        Memilih aksi dengan menambahkan noise pada output actor (exploration).
        """
        # Pengurangan noise setiap beberapa iterasi (opsional).
        if self.iterasi % 2000 == 0 and self.iterasi > 0:
            self.sigma *= 0.998  

        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state_tf, training=False)[0]
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        action = action.numpy() + noise
        action = np.clip(action, -1, 1)
        return action

    def select_action_target_network(self, next_state):
        """
        Memilih aksi dari target_actor lalu menambahkan noise 
        (sesuai paper TD3: target policy smoothing).
        """
        action = self.target_actor(next_state, training=False)
        noise = tf.random.normal(
            shape=tf.shape(action),
            mean=0.0,
            stddev=self.sigma_target,
            dtype=tf.float32
        )
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        action = tf.clip_by_value(action + noise, -1.0, 1.0)
        return action

    # ===================
    # Replay Memory Utils
    # ===================
    def update_RL_memory(self, state, action, reward, next_state, done):
        """Simpan (s, a, r, s', done) ke buffer RL."""
        self.memory_B.append((state, action, reward, next_state, done))
    

    def take_RL_minibatch(self):
        """Ambil minibatch dari buffer RL."""
        minibatch = random.sample(self.memory_B, self.batch_size)
        if self.iterasi < 260:
            print("\n\n\n")
            print("minibatch: ", minibatch)
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = zip(*minibatch)
        mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(mb_dones, dtype=tf.float32)
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones

    def update_target_weights(self):
        """
        Soft update pada target network (actor & critic).
        """
        for target_param, param in zip(self.target_actor.variables, self.actor.variables):
            target_param.assign(self.tau * param + (1.0 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_param.assign(self.tau * param + (1.0 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_param.assign(self.tau * param + (1.0 - self.tau) * target_param)


    # =============
    # Save / Logging
    # =============
    def save_models(self, episode):
        """
        Menyimpan model (actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2).
        """
        actor_save_path = os.path.join(self.save_dir, f'actor_episode_{episode}.h5')
        critic_1_save_path = os.path.join(self.save_dir, f'critic_1_episode_{episode}.h5')
        critic_2_save_path = os.path.join(self.save_dir, f'critic_2_episode_{episode}.h5')
        target_actor_save_path = os.path.join(self.save_dir, f'target_actor_episode_{episode}.h5')
        target_critic_1_save_path = os.path.join(self.save_dir, f'target_critic_1_episode_{episode}.h5')
        target_critic_2_save_path = os.path.join(self.save_dir, f'target_critic_2_episode_{episode}.h5')

        self.actor.save_weights(actor_save_path)
        self.critic_1.save_weights(critic_1_save_path)
        self.critic_2.save_weights(critic_2_save_path)
        self.target_actor.save_weights(target_actor_save_path)
        self.target_critic_1.save_weights(target_critic_1_save_path)
        self.target_critic_2.save_weights(target_critic_2_save_path)

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



    # ==================
    # train TD3 original
    # ==================
    def train_actor_critic(self):
   
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.take_RL_minibatch()

        with tf.GradientTape(persistent=True) as tape:
            # Target actions
            mb_next_actions = self.select_action_target_network(mb_next_states)
            mb_next_actions = tf.reshape(mb_next_actions, (-1, self.action_dim))

            # Hitung target Q-value
            target_Q1 = self.target_critic_1([mb_next_states, mb_next_actions], training=False)
            target_Q2 = self.target_critic_2([mb_next_states, mb_next_actions], training=False)
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = tf.reshape(target_Q, (-1,))
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * target_Q
            y = tf.stop_gradient(y)

            # Hitung current Q-value
            Q1_RL = self.critic_1([mb_states, mb_actions], training=True)
            Q2_RL = self.critic_2([mb_states, mb_actions], training=True)
            Q1_RL = tf.reshape(Q1_RL, (-1,))
            Q2_RL = tf.reshape(Q2_RL, (-1,))

            # Loss Critic
            critic_1_loss = tf.reduce_mean(tf.square(y - Q1_RL))
            critic_2_loss = tf.reduce_mean(tf.square(y - Q2_RL))
        
        # Update Critic
        critic_1_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))

        del tape

        # Delayed update actor dan soft update target
        if self.iterasi % self.update_delay == 0:
            with tf.GradientTape() as tape:
                actions = self.actor(mb_states, training=True)
                # Minimizing -Q => maximizing Q
                actor_loss = -tf.reduce_mean(self.critic_1([mb_states, actions], training=True))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            del tape

            # Soft update target networks
            self.update_target_weights()
    

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
                reward_satu_episode += reward
                done = terminated or truncated

                # Simpan ke buffer
                self.update_RL_memory(state, action_RL, reward, next_state, done)
                # Update state
                state = next_state
                # Train actor dan critic
                if len(self.memory_B) > self.batch_size:
                    self.train_actor_critic()
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
