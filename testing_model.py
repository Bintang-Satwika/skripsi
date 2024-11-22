import gymnasium as gym
from tqdm import tqdm
import pygame
import time
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle
import numpy as np

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=True, wind_power=5.0, turbulence_power=0.1, render_mode='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print("Dimensi State:", state_dim, "Dimensi Aksi:", action_dim)

running = True
n_episodes = 10
#action = [0.01, 0.8]  # [(roket mati <=0, roket naik>0), (-1 : belok kiri, 0 : lurus, 1 : belok kanan)]
iterasi = 0
load_dir = 'saved_models_and_buffers_part3_sudahbagus'
episode_mulai = 600

def create_actor_network(state_dim=state_dim, action_dim=action_dim):
    state_input = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(state_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)  # Actions dalam rentang [-1, 1]
    return models.Model(inputs=state_input, outputs=output)


def create_critic_network(state_dim=state_dim, action_dim=action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    x = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=[state_input, action_input], outputs=output)

def model_creator():
    # Definisikan ulang arsitektur model
    actor = create_actor_network() 
    critic_1 = create_critic_network()  
    critic_2 = create_critic_network() 
    target_actor = create_actor_network()
    target_critic_1 = create_critic_network() 
    target_critic_2 = create_critic_network() 
    return actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2

actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2 = model_creator()

def load_models(episode, dir_loc=load_dir):
    actor_load_path = os.path.join(dir_loc, f'actor_weights_episode_{episode}.h5')
    critic_1_load_path = os.path.join(dir_loc, f'critic_1_weights_episode_{episode}.h5')
    critic_2_load_path = os.path.join(dir_loc, f'critic_2_weights_episode_{episode}.h5')
    target_actor_load_path = os.path.join(dir_loc, f'target_actor_weights_episode_{episode}.h5')
    target_critic_1_load_path = os.path.join(dir_loc, f'target_critic_1_weights_episode_{episode}.h5')
    target_critic_2_load_path = os.path.join(dir_loc, f'target_critic_2_weights_episode_{episode}.h5')

    # Check if files exist
    for path in [actor_load_path, critic_1_load_path, critic_2_load_path, target_actor_load_path, target_critic_1_load_path, target_critic_2_load_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load weights
    actor.load_weights(actor_load_path)
    critic_1.load_weights(critic_1_load_path)
    critic_2.load_weights(critic_2_load_path)
    target_actor.load_weights(target_actor_load_path)
    target_critic_1.load_weights(target_critic_1_load_path)
    target_critic_2.load_weights(target_critic_2_load_path)

    print(f'Models loaded from episode {episode}')
    return actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2

actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2 = load_models(episode_mulai, dir_loc=load_dir)
bias_output_actor= actor.layers[-1].get_weights()[-1]
print("bias_output_actor: ", bias_output_actor)


def select_action(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    action = actor(tf.expand_dims(state, axis=0), training=False)[0]
    action = np.clip(action, -1, 1)
  #  print("action.shape", action.shape)
    return action


for episode in tqdm(range(1,n_episodes+1)):

    state, info = env.reset()
    reward_satu_episode=0
    done = False
    print("\nepisode:", episode, " mulai.")
    iterasi=0
    # play episode
    while done is False and iterasi < 230:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break
        action = select_action(state)
        env.render()
        time.sleep(0.01)
        next_state, reward, terminated, truncated, info = env.step(action)
        # update if the environment is done and the current obs
        done = terminated or truncated
        state = next_state
        iterasi += 1
        reward_satu_episode += reward
    print("\niterasi:", iterasi)
    print(f"Episode: {episode}, Total Reward:", reward_satu_episode)

pygame.display.quit()
pygame.quit()
env.close()