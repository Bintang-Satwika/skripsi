import gymnasium as gym
from tqdm import tqdm
import random
import pygame
import time
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from collections import deque

tf.keras.backend.set_floatx('float32')
save_dir = 'saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

env = gym.make(
    "LunarLander-v3",
    continuous=True,
    gravity=-10,
    enable_wind=True,
    wind_power=0,
    turbulence_power=0,
    render_mode='human'
)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print(state_dim, action_dim)

queue_length = 15
Quen = deque(maxlen=queue_length)
Quen.append(0)
buffer_length=50*1000
memory_B=deque(maxlen=int(buffer_length))
memory_B_human=deque(maxlen=int(buffer_length))

running = True
n_episodes = 300
#action=[0.01,0.8] # [(roket mati <=0, roket naik>0), (-1 : belok kiri, 0 : lurus, 1 : belok kanan)]
iterasi=0
batch_size=256
learning_rate=0.0001
sigma=0.4
reward_seluruh_episode = 0

def create_actor_network(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim))
    x = layers.Dense(256, activation='relu')(state_input)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(action_dim, activation='tanh')(x)  # Actions dalam rentang [-1, 1]
    return models.Model(inputs=state_input, outputs=output)

actor = create_actor_network(state_dim, action_dim)

def create_critic_network(state_dim, action_dim):
    inputs_critic = [layers.Input(shape=(state_dim)),layers.Input(shape=(action_dim))]
    x = layers.Concatenate(axis=-1)(inputs_critic)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs= inputs_critic, outputs=output)

critic_1=create_critic_network(state_dim, action_dim)
critic_2=create_critic_network(state_dim, action_dim)


actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

target_actor = create_actor_network(state_dim, action_dim)
target_actor.set_weights(actor.get_weights())

target_critic_1 = create_critic_network(state_dim, action_dim)
target_critic_1.set_weights(critic_1.get_weights())
target_critic_2 = create_critic_network(state_dim, action_dim)
target_critic_2.set_weights(critic_2.get_weights())

def select_action(state, step=0):
    global sigma
    if step%2000==0:
        sigma=sigma*0.998
    action = actor(state, training=False)
    noise = tf.random.normal(shape=tf.shape(action), mean=0.0, stddev=sigma, dtype=tf.float32)
    action = action + noise
    action = tf.clip_by_value(action, clip_value_min=-1, clip_value_max=1)
    return action




def compute_I(critic_1, critic_2, state, action):
    #print("COMPUTE I")
    """Menghitung perbedaan nilai Q dari dua critic networks."""
    q1 = critic_1([state, action], training=False).numpy()[0][0]
    #print("q1: ",q1)
    q2 = critic_2([state, action], training=False).numpy()[0][0]
    Is=abs(q1-q2)
    return Is

def apakah_tanya_manusia(Is, Quen, reward_seluruh_episode, actor_action, reward_max=200, th=5*2.718**(-3)):

    if Is> np.max(Quen) and reward_seluruh_episode < reward_max/th:
        print("milih aksi manusia")
        human_action= np.random.uniform(-1, 1, size=(1,action_dim))   # Random action
        return human_action, True
    else:
        #print("milih actor machine")
        return actor_action, False
    
def update_memory(state, action, reward, next_state, apakah_manusia):
    global memory_B, memory_B_human
    if apakah_manusia:
        memory_B_human.append((state[0], action[0]))
    else:
        memory_B.append((state[0], action[0], reward, next_state))


def select_action_target_network(next_state):
    action = target_actor(next_state, training=False)
    noise = tf.random.normal(shape=tf.shape(action), mean=0.0, stddev=0.2, dtype=tf.float32)
    noise = tf.clip_by_value(noise, clip_value_min=-0.5, clip_value_max=0.5)  # Sesuaikan nilai noise_clip
    action = action + noise
    action = tf.clip_by_value(action, clip_value_min=-1, clip_value_max=1)
    return action


def take_minibatch_machine(memory_B, batch_size=256):
    
    #print("Take Minibatch")
    minibatch = random.sample(memory_B, batch_size)
    
    mb_states, mb_actions, mb_rewards, mb_next_states = zip(*minibatch)
    
    # Konversi ke tensor
    mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
    mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
    mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
    mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
    


    return mb_states, mb_actions, mb_rewards, mb_next_states



@tf.function(reduce_retracing=True)
def TD_error(discount_factor, mb_states, mb_actions, mb_rewards, mb_next_states):
   
    global critic_1, critic_1_optimizer, critic_2, critic_2_optimizer
    mb_next_actions = select_action_target_network(mb_next_states)
    mb_next_actions= tf.reshape(mb_next_actions, (-1,2))
  
    # Komputasi Q target
    Q1_target = target_critic_1([mb_next_states, mb_next_actions], training=False)
    Q2_target = target_critic_2([mb_next_states, mb_next_actions], training=False)
    y1 = mb_rewards + discount_factor * Q1_target
    y2 = mb_rewards + discount_factor * Q2_target
    y_min = tf.minimum(y1, y2)
    y_min = tf.stop_gradient(y_min)
    
    # gradient  descent critic_1
    with tf.GradientTape() as tape1:
        Q1 = critic_1([mb_states, mb_actions], training=True)
        loss_1 = tf.reduce_mean(tf.square(y_min - Q1))
    grads_critic_1 = tape1.gradient(loss_1, critic_1.trainable_variables)
    critic_1_optimizer.apply_gradients(zip(grads_critic_1, critic_1.trainable_variables))
    
    # gradient  descent critic_2
    with tf.GradientTape() as tape2:
        Q2 = critic_2([mb_states, mb_actions], training=True)
        loss_2 = tf.reduce_mean(tf.square(y_min - Q2))
    grads_critic_2 = tape2.gradient(loss_2, critic_2.trainable_variables)
    critic_2_optimizer.apply_gradients(zip(grads_critic_2, critic_2.trainable_variables))
    
    #print("mini batch Loss critic_1:", loss_1, "mini batch Loss critic_2:", loss_2)



def take_minibatch_human(memory_B_human, batch_size=256):
    minibatch = random.sample(memory_B_human, batch_size)
    
    mb_states_human, mb_actions_human= zip(*minibatch)

    # Konversi ke tensor
    mb_states_human = tf.convert_to_tensor(mb_states_human, dtype=tf.float32)
    mb_actions_human = tf.convert_to_tensor(mb_actions_human, dtype=tf.float32)
   

    return mb_states_human, mb_actions_human

#@tf.function(reduce_retracing=True)
def advantage_loss(mb_states_human, mb_actions_human, step):
    global critic_1, critic_1_optimizer, critic_2, critic_2_optimizer
    with tf.GradientTape() as tape1:
        mb_actions_policy = select_action(mb_states_human, step=step)
        Q1_machine = critic_1([mb_states_human, mb_actions_policy], training=True)
        Q1_human = critic_1([mb_states_human, mb_actions_human], training=True)
        loss_1 = tf.reduce_mean(Q1_human - Q1_machine)

    grads_critic_1 = tape1.gradient(loss_1, critic_1.trainable_variables)
    critic_1_optimizer.apply_gradients(zip(grads_critic_1, critic_1.trainable_variables))
    #del tape1

    with tf.GradientTape() as tape2:
        mb_actions_policy = select_action(mb_states_human, step=step)
        Q2_machine = critic_2([mb_states_human, mb_actions_policy], training=True)
        Q2_human = critic_2([mb_states_human, mb_actions_human], training=True)
        loss_2 = tf.reduce_mean(Q2_human - Q2_machine)

    grads_critic_2 = tape2.gradient(loss_2, critic_2.trainable_variables)
    critic_2_optimizer.apply_gradients(zip(grads_critic_2, critic_2.trainable_variables))




@tf.function(reduce_retracing=True)
def gradient_ascent_actor(mb_states):
    global actor, actor_optimizer
    with tf.GradientTape() as tape:
        actions = actor(mb_states, training=True)
        Q_machine = critic_1([mb_states, actions], training=True)
        loss = - tf.reduce_mean(Q_machine) # ada minus karena gradient ascent bukan descent
    grads_actor = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads_actor, actor.trainable_variables))


def update_target_weights(tau=0.005):
    #print("\n\nUpdate Target Weights")
    global target_actor, target_critic_1, target_critic_2

    # Update target actor weights
    for target_weights, weights in zip(target_actor.trainable_weights, actor.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)

    # Update target critic_1 weights
    for target_weights, weights in zip(target_critic_1.trainable_weights, critic_1.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)

    # Update target critic_2 weights
    for target_weights, weights in zip(target_critic_2.trainable_weights, critic_2.trainable_weights):
        target_weights.assign(tau * weights + (1 - tau) * target_weights)
    

    


for episode in tqdm(range(1, n_episodes)):

    state, info = env.reset()

    done = False
    reward_satu_episode = 0
    # play episode
    while not done:
        iterasi+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break


        state=np.reshape(state, (-1, 8))
        action=select_action(state, step=iterasi)
        action=np.reshape(action, (-1, 2))
        env.render()
        time.sleep(0.01)
        next_state, reward, terminated, truncated, info = env.step(action[0])
        reward_satu_episode +=reward
        reward_seluruh_episode +=reward_satu_episode
        
        # TANPA MANUSIA
        update_memory(state=state, action=action, reward=reward, 
              next_state=next_state, apakah_manusia=False)
        
        if len(memory_B) > 256:
            mb_states, mb_actions, mb_rewards, mb_next_states=take_minibatch_machine(memory_B, batch_size=batch_size)
            TD_error(discount_factor=0.99, mb_states=mb_states, mb_actions=mb_actions, mb_rewards=mb_rewards, mb_next_states=mb_next_states) 
            if iterasi %2 == 0:
                #gradient_descent_actor(mb_states)
                pass
            if iterasi % 5 == 0:
                #print("\niterasi: ",iterasi)
                gradient_ascent_actor(mb_states)
                update_target_weights(tau=0.005)
                # Cetak bias dari layer output model target_critic_1
                output_layer_weights = target_critic_1.layers[-1].get_weights()
                output_layer_bias = output_layer_weights[1]  # Bias adalah elemen kedua
                #print("Output Layer Bias of target_critic_1:", output_layer_bias)
        ############################################################################################################


        # update if the environment is done
        done = terminated or truncated
        state= next_state 
        if done:
            break
        ############################################################################################################

    if episode % 20 == 0:
        # Tentukan path untuk menyimpan model
        actor_save_path = os.path.join(save_dir, f'actor_episode_{episode}.h5')
        critic_1_save_path = os.path.join(save_dir, f'critic_1_episode_{episode}.h5')
        critic_2_save_path = os.path.join(save_dir, f'critic_2_episode_{episode}.h5')
        target_actor_save_path = os.path.join(save_dir, f'target_actor_episode_{episode}.h5')
        target_critic_1_save_path = os.path.join(save_dir, f'target_critic_1_episode_{episode}.h5')
        target_critic_2_save_path = os.path.join(save_dir, f'target_critic_2_episode_{episode}.h5')

        # Menyimpan model
        actor.save(actor_save_path)
        critic_1.save(critic_1_save_path)
        critic_2.save(critic_2_save_path)
        target_actor.save(target_actor_save_path)
        target_critic_1.save(target_critic_1_save_path)
        target_critic_2.save(target_critic_2_save_path)

        print(f'Models saved at episode {episode}')
    print("\niterasi: ",iterasi)
    print(f"Episode: {episode}, reward_satu_episode: {reward_satu_episode}")

pygame.display.quit()
pygame.quit()
env.close()
