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
from TD3_model import TD3Loader
from pynput import keyboard

tf.keras.backend.set_floatx('float32')

class TD3Agent:
    def __init__(
        self,
        env_name="LunarLander-v3",
        n_episodes=400,
        save_every_episode=20,
        buffer_length=100000,
        human_buffer_length=100000,
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
        render_mode= None,
        imitation_episodes=40,
        hitl_episodes=60,
    ):
        """
        Inisialisasi agent TD3 dan hyperparameter.
        """

        # Direktori untuk menyimpan model dan buffer
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = 'HG_saved_models_3_part2'
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
        self.last_episode = 0
        self.imitation_episodes=imitation_episodes
        self.hitl_episodes = hitl_episodes
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
        self.count_human_help={}
        self.human_help = False

        # Replay buffer
        #self.memory_B = deque(maxlen=int(buffer_length))
        #self.memory_B_human = deque(maxlen=int(human_buffer_length))
        self.Quen = deque(maxlen=self.n)
        self.reward_max = 250


        self.episode_load= 200
        load_parameter_dir = os.path.join(current_dir, 'HG_saved_models_3_part1')
        replay_buffer_path = os.path.join(load_parameter_dir, f'replay_buffer_episode_{self.episode_load}.pkl')

        with open(replay_buffer_path , 'rb') as f:
            self.memory_B = pickle.load(f)
        print('len(self.memory_B):', len(self.memory_B))

        replay_buffer_human_path = os.path.join(load_parameter_dir, f'replay_buffer_human_episode_{self.episode_load}.pkl')
        with open(replay_buffer_human_path , 'rb') as f:
            self.memory_B_human = pickle.load(f)
        print('len(self.memory_B_human):', len(self.memory_B_human))

        td3loader = TD3Loader(load_dir=load_parameter_dir)
        self.actor, self.target_actor, self.critic_1, self.target_critic_1, self.critic_2, self.target_critic_2 = td3loader.load_models(episode=self.episode_load)


        total_iterasi_sebelum= self.episode_load*1000
        for index in range(1, total_iterasi_sebelum+1):
            if index % 2000 == 0:
                self.sigma *= 0.998
        print('self.sigma:', self.sigma)
        # Optimizer
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)


        # --------------------------------------
        # Variabel untuk menangani keyboard
        # --------------------------------------
        self.pressed_keys = set()  # set penampung tombol yg sedang ditekan
        self.stop_training = False  # jika True => hentikan loop
        self.listener = None  # akan diisi instance keyboard.Listener


    # ================================
    # Bagian jaringan Actor & Critic
    # ================================
    # def create_actor_network(self):
    #     """
    #     Membuat model Actor dengan output di [-1, 1] untuk setiap dimensi aksi.
    #     """
    #     state_input = layers.Input(shape=(self.state_dim,))
    #     x = layers.Dense(256, activation='relu')(state_input)
    #     x = layers.Dense(256, activation='relu')(x)
    #     output = layers.Dense(self.action_dim, activation='tanh')(x)
    #     model = models.Model(inputs=state_input, outputs=output)
    #     return model

    # def create_critic_network(self):
    #     """
    #     Membuat model Critic (Q-network). 
    #     Input: state dan action, output: Q-value.
    #     """
    #     state_input = layers.Input(shape=(self.state_dim,))
    #     action_input = layers.Input(shape=(self.action_dim,))
    #     x = layers.Concatenate()([state_input, action_input])
    #     x = layers.Dense(256, activation='relu')(x)
    #     x = layers.Dense(256, activation='relu')(x)
    #     output = layers.Dense(1)(x)
    #     model = models.Model(inputs=[state_input, action_input], outputs=output)
    #     return model

    # =============================
    # Bagian Keyboard dengan pynput
    # =============================
    def on_press(self, key):
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
    
    def update_human_memory(self, state, action, reward):
        """Simpan (s, a) dari aksi manusia ke buffer terpisah."""
        self.memory_B_human.append((state, action, reward))

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


    def take_human_guided_minibatch(self):
        """Ambil minibatch dari buffer manusia."""
        minibatch = random.sample(self.memory_B_human, self.batch_size)
        mb_states_human, mb_actions_human, mb_rewards_human = zip(*minibatch)
        mb_states_human = tf.convert_to_tensor(mb_states_human, dtype=tf.float32)
        mb_actions_human = tf.convert_to_tensor(mb_actions_human, dtype=tf.float32)
        mb_rewards_human = tf.convert_to_tensor(mb_rewards_human, dtype=tf.float32)
        return mb_states_human, mb_actions_human, mb_rewards_human


    # ==================
    # Bagian Update TD3
    # ==================

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
            replay_filename_human = os.path.join(self.save_dir, f'replay_buffer_human_episode_{episode}.pkl')
            with open(replay_filename_human, 'wb') as f:
                pickle.dump(self.memory_B_human, f)
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

        # # Simpan count_ask_human setiap episode
        # count_filename = os.path.join(self.save_dir, 'A_count_ask_human.json')
        # if os.path.exists(count_filename):
        #     # Jika file sudah ada, muat isinya
        #     with open(count_filename, 'r') as f:
        #         existing_count= json.load(f)
        # else:
        #     existing_count = {}
        # # Updatebaru
        # existing_count[episode] = self.count_ask_human[episode]
        # with open(count_filename, 'w') as f:
        #     json.dump(existing_count, f, indent=4)
        # print(f'A_count_ask_human saved to {count_filename}')

        # Simpan count_human_help setiap episode
        count_filename = os.path.join(self.save_dir, 'A_count_human_help.json')
        if os.path.exists(count_filename):
            # Jika file sudah ada, muat isinya
            with open(count_filename, 'r') as f:
                existing_count= json.load(f)
        else:
            existing_count = {}
        # Updatebaru
        existing_count[episode] = self.count_human_help[episode]
        with open(count_filename, 'w') as f:
            json.dump(existing_count, f, indent=4)
        print(f'A_count_human_help saved to {count_filename}')


    # =========================================
    # Fungsi bantuan untuk HITL & Input Keyboard
    # =========================================
    def rule_based_policy(self, state):
        """
        state = [x, y, vx, vy, angle, angularVel, legContact1, legContact2]
        
        Return nilai aksi (0, 1, 2, atau 3) berdasarkan aturan sederhana:
        - 0: diam
        - 1: engine kiri
        - 2: engine utama (ke atas)
        - 3: engine kanan
        """
        x, y, vx, vy, angle, angularVel, leg1, leg2 = state

        # 1. Tentukan sudut target (angle_targ) agar lander cenderung 'menghadap' titik (0,0).
        #    Misal kita memanfaatkan X (posisi horizontal) dan kecepatan horizontal untuk memberi sinyal seberapa
        #    besar sudut yg kita inginkan.
        angle_targ = 0.5 * x + 1.0 * vx
        #  Batasi agar tidak terlalu ekstrem, misal ±0.4 rad (~23 derajat)
        angle_targ = np.clip(angle_targ, -0.4, 0.4)

        # 2. Tentukan ketinggian target (hover_targ) agar semakin jauh dari pusat secara horizontal,
        #    kita sedikit "hover" lebih tinggi. Ini hanya salah satu strategi agar lander mengoreksi lebih cepat.
        hover_targ = 0.55 * abs(x)
        #  Hover ini kemudian dibandingkan dengan y dan vy.

        # 3. Hitung error sudut (angle) dan error ketinggian/kecepatan (hover).
        angle_error = angle_targ - angle
        angle_todo  = 0.5 * angle_error - 1.0 * angularVel

        hover_error = hover_targ - y
        hover_todo  = 0.5 * hover_error - 0.5 * vy

        # 4. Jika kaki sudah menyentuh tanah, lebih baik jangan terlalu agresif mengoreksi sudut lagi.
        #    Cukup kurangi kecepatan jatuh (vy).
        if leg1 == 1.0 or leg2 == 1.0:
            angle_todo = 0.0
            hover_todo = -0.5 * vy  # sekadar mengurangi benturan

        # 5. Terjemahkan "angle_todo" dan "hover_todo" menjadi aksi diskret:
        #    - Aksi 2 (engine utama) jika kita butuh dorongan ke atas lebih besar daripada dorongan untuk memutar.
        #    - Aksi 1 (engine kiri)  jika lander harus berotasi ke kanan (angle_todo > 0).
        #    - Aksi 3 (engine kanan) jika lander harus berotasi ke kiri  (angle_todo < 0).
        #    - Aksi 0 jika tidak ada tuntutan berarti (mungkin sudah cukup stabil).
        # 
        #    Kita bisa membuat aturan sederhana:
        #    - Jika hover_todo cukup besar (> 0.05) dan melebihi |angle_todo|,
        #      kita prioritaskan menyalakan engine utama (aksi 2).
        #    - Kalau angle_todo > 0.05 => aksi 1 (engine kiri).
        #    - Kalau angle_todo < -0.05 => aksi 3 (engine kanan).
        #    - Sisanya => aksi 0 (diam).
        array_action=[0,0]
        main_threshold = 0.05
        angle_threshold = 0.05

        if (hover_todo > abs(angle_todo)) and (hover_todo > main_threshold):
            array_action[0] = 1  # engine utama
        else:
            array_action[0] = 0  # diam
            
        if angle_todo < -angle_threshold:
            array_action[1] = 1 # engine orientasi kanan (lateral kiri)
        elif angle_todo > angle_threshold:
            array_action[1]=-1 # engine orientasi kiri (lateral kanan)
        else:
            array_action[1]=0

        return array_action


    def ask_human(self, state, action, reward_satu_episode):
        """
        Mengecek kondisi untuk memutuskan apakah perlu minta bantuan manusia.
        Berdasarkan selisih Q1-Q2 dan reward episode.
        """
        action = tf.reshape(action, (-1, self.action_dim))
        state = tf.reshape(state, (-1, self.state_dim))

        # self.main_throttle = action[0][0].numpy().item()
        #self.lateral_throttle = action[0][1].numpy().item()

        Q1_actual = self.critic_1([state, action], training=False)
        Q2_actual = self.critic_2([state, action], training=False)
        Is = (Q1_actual - Q2_actual).numpy().item()

        # Jika Q1 - Q2 lebih besar dari max di self.Quen & reward masih di bawah threshold => human_help
        if len(self.Quen) > 0:
            if Is > max(self.Quen) and reward_satu_episode < self.reward_max / self.th:
                self.human_help = True
            else:
                self.human_help = False
        else:
            self.human_help = False

        self.Quen.append(Is)

    def human_in_the_loop(self, state, action, reward_satu_episode):
        """
        Memeriksa apakah perlu human-in-the-loop, lalu ambil aksi human jika perlu.
        """
        # Jika tidak lagi di intervensi, cek apakah butuh minta bantuan
        self.ask_human(state, action, reward_satu_episode)

        # Jika perlu bantuan manusia
        if self.human_help:
            # Setting throttle awal sesuai action RL terakhir
            action_human = self.rule_based_policy(state)

        else:
            action_human = None

        return action_human
    

    def train_actor_critic_human_guided(self):

        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = self.take_RL_minibatch()
        mb_states_human, mb_actions_human, mb_rewards_human     = self.take_human_guided_minibatch()
         # Hitung weight adaptif dari perbedaan reward (hindari pembagian dengan nol)
        weight_human = tf.math.divide_no_nan(
            tf.maximum(0.0, mb_rewards_human - mb_rewards),
            tf.abs(mb_rewards_human) + 1e-6
        )
        
        with tf.GradientTape(persistent=True) as tape:
            # --- Perhitungan Target Q-value ---
            # Hitung aksi target untuk next states
            mb_next_actions = self.select_action_target_network(mb_next_states)
            mb_next_actions = tf.reshape(mb_next_actions, (-1, self.action_dim))
            
            # Evaluasi Q-value target dari kedua target critic
            target_Q1 = self.target_critic_1([mb_next_states, mb_next_actions], training=False)
            target_Q2 = self.target_critic_2([mb_next_states, mb_next_actions], training=False)
            target_Q = tf.minimum(target_Q1, target_Q2)
            target_Q = tf.reshape(target_Q, (-1,))
            
            # Hitung target nilai TD
            y = tf.stop_gradient(mb_rewards + (1.0 - mb_dones) * self.gamma * target_Q)
            
            # --- Perhitungan Q-value untuk data RL ---
            Q1_RL = tf.reshape(self.critic_1([mb_states, mb_actions], training=True), (-1,))
            Q2_RL = tf.reshape(self.critic_2([mb_states, mb_actions], training=True), (-1,))
            
            # --- Perhitungan Q-value untuk data manusia ---
            # Q-value dari pasangan (state, action) demonstrasi manusia
            Q1_human = tf.reshape(self.critic_1([mb_states_human, mb_actions_human], training=True), (-1,))
            Q2_human = tf.reshape(self.critic_2([mb_states_human, mb_actions_human], training=True), (-1,))
            # Q-value dari pasangan (state, actor_predicted_action) pada data manusia
            mb_actions_predicted = self.actor(mb_states_human, training=True)
            Q1_policy = tf.reshape(self.critic_1([mb_states_human, mb_actions_predicted], training=True), (-1,))
            Q2_policy = tf.reshape(self.critic_2([mb_states_human, mb_actions_predicted], training=True), (-1,))
            
            # --- Perhitungan Advantage Loss (Loss tambahan untuk integrasi sinyal manusia) ---
            advantage_loss_1 = tf.reduce_mean(weight_human * tf.square(Q1_human - Q1_policy))
            advantage_loss_2 = tf.reduce_mean(weight_human * tf.square(Q2_human - Q2_policy))
            
            # --- Perhitungan Total Loss untuk masing-masing critic ---
            critic_1_loss = tf.reduce_mean(tf.square(y - Q1_RL)) + advantage_loss_1
            critic_2_loss = tf.reduce_mean(tf.square(y - Q2_RL)) + advantage_loss_2
        
        # --- Update parameter critic ---
        critic_1_grad = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grad = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grad, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grad, self.critic_2.trainable_variables))
        
        del tape
        # Update Actor
        if self.iterasi % self.update_delay == 0:
            with tf.GradientTape() as tape:
                actions = self.actor(mb_states, training=True)
                # Minimizing -Q => maximizing Q
                actor_RL_loss = -self.critic_1([mb_states, actions], training=True)
                mb_actions_predicted = self.actor(mb_states_human, training=True)
                behaviour_loss = tf.square(mb_actions_human - mb_actions_predicted)
                actor_human_loss= tf.tensordot(weight_human, behaviour_loss, axes=1)/self.batch_size
                total_actor_loss = tf.add(tf.reduce_mean(actor_RL_loss), tf.reduce_mean(actor_human_loss))

            actor_grad = tape.gradient(total_actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            del tape

            # Soft update target networks
            self.update_target_weights()


    # =======================
    # Training / Main Loop
    # =======================
    def imitation_learning(self, state):

        if len(self.memory_B_human) > self.batch_size :
            minibatch = random.sample(self.memory_B_human, self.batch_size)
            mb_states_human, mb_actions_human, mb_rewards_human = zip(*minibatch)
            mb_states_human = tf.convert_to_tensor(mb_states_human, dtype=tf.float32)
            mb_actions_human = tf.convert_to_tensor(mb_actions_human, dtype=tf.float32)
            mb_rewards_human = tf.convert_to_tensor(mb_rewards_human, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                mb_actions_predicted = self.actor(mb_states_human, training=True)
                behaviour_loss = tf.reduce_mean(tf.square(mb_actions_human - mb_actions_predicted))
            
            actor_grad = tape.gradient( behaviour_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            del tape

            if self.iterasi % self.update_delay == 0:
                for target_param, param in zip(self.target_actor.variables, self.actor.variables):
                    target_param.assign(self.tau * param + (1.0 - self.tau) * target_param)
        
        action_human =self.rule_based_policy(state)
        return action_human


    def main_loop(self):
        """
        Loop utama untuk melatih agent.
        """


        for episode in tqdm(range(self.episode_load+1, self.n_episodes + 1)):
            if self.stop_training:
                # Jika user menekan ESC sebelum mulai episode
                break

            state, info = self.env.reset(seed=episode)
            done = False
            reward_satu_episode = 0
            action_human = None
            action = None
            action_RL = None
            max_iterasi_episode= 0
            self.count_episode = episode
            time.sleep(0.03)
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
                            
                if episode <= self.imitation_episodes:
                    action_human = self.imitation_learning(state)

                elif self.imitation_episodes < episode <= self.imitation_episodes + self.hitl_episodes:
                    # Aksi RL  (tanpa bantuan manusia)
                    action_RL = self.select_action(state)
                    action = action_RL
                    # human_in_the_loop
                    action_human = self.human_in_the_loop(state, action_RL, reward_satu_episode)
                else:
                    # Aksi RL  (tanpa bantuan manusia)
                    action_RL = self.select_action(state)
                    action = action_RL
                    action_human=None

                
                # Gunakan aksi human bila tidak None, jika None => pakai aksi RL
                action = action_human if action_human is not None else action_RL
                next_state, reward, terminated, truncated, info = self.env.step(action)
                reward_satu_episode += reward
                done = terminated or truncated

                # Simpan ke buffer
                if action_human is not None:
                    try:
                        self.count_human_help[self.count_episode] += 1
                    except:
                        self.count_human_help[self.count_episode] = 1

                    self.update_human_memory(state, action_human, reward)
                else:
                    try:
                        self.count_human_help[self.count_episode] += 0
                    except:
                        self.count_human_help[self.count_episode] = 0
                    self.update_RL_memory(state, action_RL, reward, next_state, done)
                
                # Update state
                state = next_state

                # Train actor dan critic
                if len(self.memory_B) > self.batch_size:
                    self.train_actor_critic_human_guided()

                # Render environment
                #self.env.render()

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
