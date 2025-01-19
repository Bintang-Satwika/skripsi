import gymnasium as gym
from tqdm import tqdm
import random
import pygame
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras import layers, models
from collections import deque
from matplotlib import pyplot as plt
import json

# Biasanya bisa pakai ini untuk menonaktifkan warning keras (opsional):
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.keras.backend.set_floatx('float32')


class TD3Agent:
    def __init__(
        self,
        env_name="LunarLander-v3",
        n_episodes=10,
        save_every_episode=10,
        buffer_length=5000,
        human_buffer_length=10000,
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
        self.save_dir = 'coba1'
        os.makedirs(self.save_dir, exist_ok=True)

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

        # Inisialisasi nilai throttle
        self.main_throttle = 0.0
        self.lateral_throttle = 0.0
        self.throttle_rate = 0.1

        # Bungkus environment untuk merekam statistik episode
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
        self.memory_B_human = deque(maxlen=int(human_buffer_length))
        self.Quen = deque(maxlen=self.n)  # Menyimpan nilai Q1-Q2
        self.reward_max = 250

        # Pengaturan durasi intervensi manusia
        self.min_human_step = 100
        self.human_step_now = 0
        self.max_step_no_keyboard = 10
        self.step_no_keyboard_now = 0

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


    def select_action(self, state):
        """
        Memilih aksi dengan menambahkan noise pada output actor (exploration).
        """
        if self.iterasi % 2000 == 0 and self.iterasi > 0:
            self.sigma *= 0.998  # Contoh penurunan sigma

        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state_tf, training=False)[0].numpy()
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        action = np.clip(action + noise, -1, 1)
        return action


    def select_action_target_network(self, next_states):
        """
        Memilih aksi dari target_actor lalu menambahkan noise 
        (sesuai paper TD3: target policy smoothing).
        """
        action = self.target_actor(next_states, training=False)
        noise = tf.random.normal(
            shape=tf.shape(action),
            mean=0.0,
            stddev=self.sigma_target,
            dtype=tf.float32
        )
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        action = tf.clip_by_value(action + noise, -1.0, 1.0)
        return action


    def update_memory_rl(self, s, a, r, s_next, done):
        """Simpan transition RL ke buffer."""
        self.memory_B.append((s, a, r, s_next, done))


    def update_memory_human(self, s, a, r, s_next, done):
        """Simpan transition Human ke buffer."""
        self.memory_B_human.append((s, a, r, s_next, done))


    def take_minibatch(self, memory, batch_size):
        """Ambil minibatch dari buffer (RL/human)."""
        minibatch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        return states, actions, rewards, next_states, dones


    def critic_loss_rl(self, mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones):
        """
        Proses update critic untuk pengalaman RL (TD3 utama).
        """
        with tf.GradientTape(persistent=True) as tape:
            # Aksi target + smoothing
            mb_next_actions = self.select_action_target_network(mb_next_states)
            # Q-value dari target network
            target_Q1 = self.target_critic_1([mb_next_states, mb_next_actions], training=False)
            target_Q2 = self.target_critic_2([mb_next_states, mb_next_actions], training=False)
            target_Q = tf.minimum(target_Q1, target_Q2)

            # y = r + gamma * (1-done) * target_Q
            target_Q = tf.reshape(target_Q, (-1,))
            y = mb_rewards + (1.0 - mb_dones) * self.gamma * target_Q
            y = tf.stop_gradient(y)

            # Current Q
            current_Q1 = self.critic_1([mb_states, mb_actions], training=True)
            current_Q2 = self.critic_2([mb_states, mb_actions], training=True)
            current_Q1 = tf.reshape(current_Q1, (-1,))
            current_Q2 = tf.reshape(current_Q2, (-1,))

            critic_1_loss = tf.reduce_mean((y - current_Q1)**2)
            critic_2_loss = tf.reduce_mean((y - current_Q2)**2)

        # Backprop Critic
        critic_1_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))
        del tape


    def critic_loss_human(self, mb_states_human, mb_actions_human):
        """
        Meng-update critic berdasar data 'human'.
        (Sebagai contoh: advantage-based approach)
        """
        with tf.GradientTape(persistent=True) as tape:
            Q1_human = self.critic_1([mb_states_human, mb_actions_human], training=True)
            Q2_human = self.critic_2([mb_states_human, mb_actions_human], training=True)
            Q1_human = tf.reshape(Q1_human, (-1,))
            Q2_human = tf.reshape(Q2_human, (-1,))

            # Aksi policy
            actions_policy = self.actor(mb_states_human, training=True)
            Q1_policy = self.critic_1([mb_states_human, actions_policy], training=True)
            Q2_policy = self.critic_2([mb_states_human, actions_policy], training=True)
            Q1_policy = tf.reshape(Q1_policy, (-1,))
            Q2_policy = tf.reshape(Q2_policy, (-1,))

            # Advantage
            advantage_loss_1 = tf.reduce_mean(Q1_human - Q1_policy)
            advantage_loss_2 = tf.reduce_mean(Q2_human - Q2_policy)

        # Backprop Critic
        grad_1 = tape.gradient(advantage_loss_1, self.critic_1.trainable_variables)
        grad_2 = tape.gradient(advantage_loss_2, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(grad_1, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(grad_2, self.critic_2.trainable_variables))
        del tape


    def actor_loss_and_update_target(self, mb_states):
        """
        Delayed update actor dan soft update target networks.
        """
        if self.iterasi % self.update_delay == 0:
            with tf.GradientTape() as tape:
                actions = self.actor(mb_states, training=True)
                actor_loss = -tf.reduce_mean(self.critic_1([mb_states, actions], training=True))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Soft update target networks
            self.update_target_weights()


    def update_target_weights(self):
        """
        Lakukan soft update pada target network (actor & critic).
        """
        for t_param, param in zip(self.target_actor.variables, self.actor.variables):
            t_param.assign(self.tau * param + (1.0 - self.tau) * t_param)

        for t_param, param in zip(self.target_critic_1.variables, self.critic_1.variables):
            t_param.assign(self.tau * param + (1.0 - self.tau) * t_param)

        for t_param, param in zip(self.target_critic_2.variables, self.critic_2.variables):
            t_param.assign(self.tau * param + (1.0 - self.tau) * t_param)


    def save_models(self, episode):
        """
        Menyimpan model (actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2).
        """
        if episode % self.save_every_episode == 0:
            actor_save_path = os.path.join(self.save_dir, f'actor_episode_{episode}.h5')
            critic_1_save_path = os.path.join(self.save_dir, f'critic_1_episode_{episode}.h5')
            critic_2_save_path = os.path.join(self.save_dir, f'critic_2_episode_{episode}.h5')
            t_actor_save_path = os.path.join(self.save_dir, f'target_actor_episode_{episode}.h5')
            t_critic_1_save_path = os.path.join(self.save_dir, f'target_critic_1_episode_{episode}.h5')
            t_critic_2_save_path = os.path.join(self.save_dir, f'target_critic_2_episode_{episode}.h5')

            self.actor.save_weights(actor_save_path)
            self.critic_1.save_weights(critic_1_save_path)
            self.critic_2.save_weights(critic_2_save_path)
            self.target_actor.save_weights(t_actor_save_path)
            self.target_critic_1.save_weights(t_critic_1_save_path)
            self.target_critic_2.save_weights(t_critic_2_save_path)

            print(f'[SAVE] Models at episode {episode}')


    def save_replay_buffer_and_rewards(self, episode):
        """
        Menyimpan replay buffer ke file pickle dan reward kumulatif ke file JSON.
        """
        if episode % self.save_every_episode == 0:
            replay_filename = os.path.join(self.save_dir, f'replay_buffer_episode_{episode}.pkl')
            with open(replay_filename, 'wb') as f:
                pickle.dump(self.memory_B, f)
            print(f'[SAVE] Replay buffer to {replay_filename}')

        rewards_filename = os.path.join(self.save_dir, 'A_cumulative_rewards.json')
        # Muat jika sudah ada
        if os.path.exists(rewards_filename):
            with open(rewards_filename, 'r') as f:
                existing_rewards = json.load(f)
        else:
            existing_rewards = {}

        existing_rewards[episode] = self.cumulative_reward_episode[episode]
        with open(rewards_filename, 'w') as f:
            json.dump(existing_rewards, f, indent=4)
        print(f'[SAVE] Episode Reward to {rewards_filename}')


    def get_action(self, keys):
        """
        Mengonversi input keyboard menjadi aksi kontinu untuk LunarLander.
        Menambahkan 'toleransi' step_no_keyboard_now saat tombol dilepas.
        """
        dummy_main = False
        dummy_lateral = False
        dummy_main_2 = False
        dummy_lateral_2 = False

        # Tombol throttle utama (naik/turun)
        if keys[pygame.K_w]:
            self.main_throttle = min(self.main_throttle + self.throttle_rate, 1.0)
        elif keys[pygame.K_s]:
            self.main_throttle = max(self.main_throttle - self.throttle_rate, -1.0)
        else:
            # Jika tidak menekan W/S dan masih ada sisa toleransi step_no_keyboard_now
            if self.step_no_keyboard_now > 0:
                if self.main_throttle > 0:
                    self.main_throttle = max(0.0, self.main_throttle - self.throttle_rate)
                else:
                    self.main_throttle = min(0.0, self.main_throttle + self.throttle_rate)
                dummy_main = True
            else:
                dummy_main_2 = True

        # Tombol throttle lateral (kiri/kanan)
        if keys[pygame.K_d]:
            self.lateral_throttle = max(self.lateral_throttle - self.throttle_rate, -1.0)
        elif keys[pygame.K_a]:
            self.lateral_throttle = min(self.lateral_throttle + self.throttle_rate, 1.0)
        else:
            if self.step_no_keyboard_now > 0:
                if self.lateral_throttle > 0:
                    self.lateral_throttle = max(0.0, self.lateral_throttle - self.throttle_rate)
                else:
                    self.lateral_throttle = min(0.0, self.lateral_throttle + self.throttle_rate)
                dummy_lateral = True
            else:
                dummy_lateral_2 = True

        # Jika sama sekali tidak ada tombol ditekan dan toleransi habis => None
        if dummy_main_2 and dummy_lateral_2:
            return None

        # Kalau user masih "idle" tapi belum melebihi step_no_keyboard
        if dummy_main and dummy_lateral:
            self.step_no_keyboard_now = max(0, self.step_no_keyboard_now - 1)
        else:
            self.step_no_keyboard_now = self.max_step_no_keyboard

        return np.array([self.main_throttle, self.lateral_throttle], dtype=np.float32)


    def ask_human(self, state, action, reward_satu_episode):
        """
        Mengecek kondisi untuk memutuskan apakah perlu minta bantuan manusia.
        """
        s = tf.reshape(state, (1, -1))
        a = tf.reshape(action, (1, -1))

        # Meniru throttle RL di awal
        self.main_throttle = a[0][0].numpy()
        self.lateral_throttle = a[0][1].numpy()

        Q1_actual = self.critic_1([s, a], training=False)
        Q2_actual = self.critic_2([s, a], training=False)
        Is = (Q1_actual - Q2_actual).numpy().item()

        # Syarat butuh human
        if self.Quen and (Is > max(self.Quen)) and (reward_satu_episode < self.reward_max / self.th):
            self.human_help = True
            self.human_step_now = self.min_human_step
        else:
            self.human_help = False

        self.Quen.append(Is)


    def train(self):
        """
        Loop utama untuk melatih agent.
        """
        running = True

        for episode in tqdm(range(1, self.n_episodes + 1)):
            state, _ = self.env.reset()
            done = False
            reward_satu_episode = 0.0
            action_human = None
            action_rl = None

            # Reset hitungan step
            self.human_step_now = 0
            self.step_no_keyboard_now = 0

            while not done:
                self.iterasi += 1

                # Event handler pygame (menutup window)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
                if not running:
                    break

                # Jika human tidak aktif atau toleransi step keyboard habis
                # maka ambil aksi RL.
                if self.human_step_now == 0 or self.step_no_keyboard_now == 0 or action_human is None:
                    action_rl = self.select_action(state)

                # Cek syarat butuh human-in-the-loop
                if self.human_step_now == 0:
                    self.ask_human(state, action_rl, reward_satu_episode)

                # Jika butuh human
                if self.human_help:
                    # Throttle pertama meniru RL
                    self.main_throttle = action_rl[0]
                    self.lateral_throttle = action_rl[1]

                    keys = pygame.key.get_pressed()
                    action_human = self.get_action(keys)
                    print("action_human", action_human)
                    # Jika step_no_keyboard_now == 0 => user lepas lebih lama => None => RL.
                    if self.step_no_keyboard_now == 0:
                        action_human = None

                    self.human_step_now = max(0, self.human_step_now - 1)
                else:
                    action_human = None

                # Tentukan aksi akhir
                final_action = action_human if action_human is not None else action_rl

                # Jalankan di environment
                next_state, r, terminated, truncated, _ = self.env.step(final_action)
                reward_satu_episode += r
                done = terminated or truncated

                # Simpan ke buffer
                if action_human is not None:
                    self.update_memory_human(state, action_human, r, next_state, done)
                else:
                    self.update_memory_rl(state, action_rl, r, next_state, done)

                state = next_state

                # Update network jika buffer sudah cukup
                if len(self.memory_B) > self.batch_size:
                    # 1. Update Critic RL
                    mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = \
                        self.take_minibatch(self.memory_B, self.batch_size)
                    self.critic_loss_rl(mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones)

                    # 2. Update Critic Human (jika buffer cukup)
                    if len(self.memory_B_human) > self.batch_size:
                        mb_s_h, mb_a_h, _, _, _ = self.take_minibatch(self.memory_B_human, self.batch_size)
                        self.critic_loss_human(mb_s_h, mb_a_h)

                    # 3. Update Actor dan Target
                    self.actor_loss_and_update_target(mb_states)

                # **Opsional**: Render environment (mengurangi performa)
                self.env.render()

            print(f"Episode: {episode} | Reward: {reward_satu_episode:.2f} | Iterasi: {self.iterasi}")

            # Simpan reward kumulatif
            self.cumulative_reward_episode[episode] = reward_satu_episode
            self.save_replay_buffer_and_rewards(episode)
            self.save_models(episode)

            if not running:
                break

        # Tutup pygame dan environment
        pygame.display.quit()
        pygame.quit()
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
    agent.train()
    agent.plot_results()
