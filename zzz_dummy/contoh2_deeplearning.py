import pickle
import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models


state_dim=8
action_dim=2
load_dir = 'zzz_dummy/saved_models_and_buffers_part3_sudahbagus'
episode_mulai = 600
batch_size=3

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


def load_replay_buffer(filename):
    with open(filename, 'rb') as f:
        memory_B = pickle.load(f)
        print(len(memory_B))
    print(f'Replay buffer loaded from {filename}')
    return memory_B

#replay_buffer_path = os.path.join('D:\KULIAH\skripsi\coba_2a', f'replay_buffer_episode_{10}.pkl')
#memory_B = load_replay_buffer(replay_buffer_path)
memory_B = load_replay_buffer('zzz_dummy/human_guided_3/replay_buffer_episode_10.pkl')
def take_RL_minibatch(seed=0):
    """Ambil minibatch dari buffer RL."""
    random.seed(seed)
    minibatch = random.sample(memory_B, batch_size)
    mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = zip(*minibatch)
    mb_states = tf.convert_to_tensor(mb_states, dtype=tf.float32)
    mb_actions = tf.convert_to_tensor(mb_actions, dtype=tf.float32)
    mb_rewards = tf.convert_to_tensor(mb_rewards, dtype=tf.float32)
    mb_next_states = tf.convert_to_tensor(mb_next_states, dtype=tf.float32)
    mb_dones = tf.convert_to_tensor(mb_dones, dtype=tf.float32)
    return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones


mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = take_RL_minibatch(seed=0)
print("mb_rewards: ", mb_rewards)

mb_states_human, mb_actions_human, mb_rewards_human, mb_next_states_human, mb_dones_human = take_RL_minibatch(seed=1)
print("mb_rewards_human: ", mb_rewards_human)
weight_human= tf.divide(tf.maximum(0, mb_rewards_human - mb_rewards), tf.abs(mb_rewards_human)+ 1e-6 )
print("weight_human: ", weight_human)


update_delay=2
noise_clip=0.5
gamma=0.99
def select_action_target_network(next_state):
        """
        Memilih aksi dari target_actor lalu menambahkan noise 
        (sesuai paper TD3: target policy smoothing).
        """
        action = target_actor(next_state, training=False)
        noise = tf.random.normal(
            shape=tf.shape(action),
            mean=0.0,
            stddev=0.2,
            dtype=tf.float32
        )
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        action = tf.clip_by_value(action + noise, -1.0, 1.0)
        return action

print("\n")
with tf.GradientTape(persistent=True) as tape:
    # Target actions
    mb_next_actions = select_action_target_network(mb_next_states)
    mb_next_actions = tf.reshape(mb_next_actions, (-1, action_dim))

    # Hitung target Q-value
    target_Q1 = target_critic_1([mb_next_states, mb_next_actions], training=False)
    target_Q2 = target_critic_2([mb_next_states, mb_next_actions], training=False)
    print("target_Q1: ", target_Q1)
    print("target_Q2: ", target_Q2)
    target_Q = tf.minimum(target_Q1, target_Q2) 
    print("target_Q: ", target_Q)
    target_Q = tf.reshape(target_Q, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    print("target_Q: ", target_Q)
    print("mb_dones: ", mb_dones)
    print("mb_rewards: ", mb_rewards)
    y = mb_rewards + (1.0 - mb_dones) * gamma * target_Q
   # y = tf.reshape(y, (-1,)) TIDAK PERLU RESHAPE karena sudah 1D
    y = tf.stop_gradient(y)
    print("y: ", y)

    # # Hitung current Q-value
    Q1_RL = critic_1([mb_states, mb_actions], training=True)
    Q2_RL = critic_2([mb_states, mb_actions], training=True)
    print("Q1_RL: ", Q1_RL)
    Q1_RL = tf.reshape(Q1_RL, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    Q2_RL = tf.reshape(Q2_RL, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    print("Q1_RL: ", Q1_RL)

    # # Q-value Dari state + action manusia
    Q1_human = critic_1([mb_states_human, mb_actions_human], training=True)
    Q2_human = critic_2([mb_states_human, mb_actions_human], training=True)
    print("Q1_human: ", Q1_human)
    Q1_human = tf.reshape(Q1_human, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    print("Q1_human: ", Q1_human)
    Q2_human = tf.reshape(Q2_human, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D

    # # Q-value dari prediksi action oleh actor
    mb_actions_predicted= actor(mb_states_human, training=True)
    print("mb_actions_predicted: ", mb_actions_predicted)
    Q1_policy = critic_1([mb_states_human,  mb_actions_predicted], training=True)
    Q2_policy = critic_2([mb_states_human,  mb_actions_predicted], training=True)
    print("Q2_policy: ", Q2_policy)
    Q1_policy = tf.reshape(Q1_policy, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    Q2_policy = tf.reshape(Q2_policy, (-1,)) #  PERLU RESHAPE dari 2D jadi 1D
    print("Q2_policy: ", Q2_policy)

    print("\n")
    print("Q1_human - Q1_policy:", Q1_human - Q1_policy)
    print("Q2_human - Q2_policy:", Q2_human - Q2_policy)
    print("weight_human: ", weight_human)
    # Advantage loss 
    advantage_loss_1 = tf.tensordot(weight_human, tf.square(Q1_human - Q1_policy), axes=1)/batch_size
    advantage_loss_2 = tf.tensordot(weight_human, tf.square(Q2_human - Q2_policy), axes=1)/batch_size
    print("advantage_loss_1: ", advantage_loss_1)
    print("advantage_loss_2: ", advantage_loss_2)

    # Loss Critic
    print("\n")
    print("y: ", y)
    print("Q1_RL: ", Q1_RL)
    critic_1_loss = tf.add(tf.reduce_mean(tf.square(y - Q1_RL)), advantage_loss_1)
    critic_2_loss = tf.add(tf.reduce_mean(tf.square(y - Q2_RL)), advantage_loss_2)
    print("critic_1_loss: ", critic_1_loss)
    print("critic_2_loss: ", critic_2_loss)

# # Update Critic
# critic_1_grad = tape.gradient(critic_1_loss, critic_1.trainable_variables)
# critic_2_grad = tape.gradient(critic_2_loss, critic_2.trainable_variables)
# critic_1_optimizer.apply_gradients(zip(critic_1_grad, critic_1.trainable_variables))
# critic_2_optimizer.apply_gradients(zip(critic_2_grad, critic_2.trainable_variables))
del tape

print("\n")
# Update Actor
with tf.GradientTape() as tape:
    actions = actor(mb_states, training=True)
    #print("actions: ", actions)
    # Minimizing -Q => maximizing Q
    actor_RL_loss = -critic_1([mb_states, actions], training=True)
    print("actor_RL_loss: ", actor_RL_loss)
    mb_actions_predicted = actor(mb_states_human, training=True)
    print("mb_actions_predicted: ", mb_actions_predicted)
    print("mb_actions_human: ", mb_actions_human)
    behaviour_loss = tf.square(mb_actions_human - mb_actions_predicted)
    print("behaviour_loss: ", behaviour_loss)
    print("weight_human: ", weight_human)
    actor_human_loss= tf.tensordot(weight_human, behaviour_loss, axes=1)/batch_size
    print("actor_human_loss: ", actor_human_loss)
    print("tf.reduce_mean(actor_RL_loss): ", tf.reduce_mean(actor_RL_loss))
    print("tf.reduce_mean(actor_human_loss): ", tf.reduce_mean(actor_human_loss))
    total_actor_loss = tf.add(tf.reduce_mean(actor_RL_loss), tf.reduce_mean(actor_human_loss))
    print("total_actor_loss: ", total_actor_loss)

# actor_grad = tape.gradient(total_actor_loss, actor.trainable_variables)
# actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))
del tape

def take_last(memory_B):
    """Ambil 1 sample terakhir dari buffer manusia."""
    #minibatch = random.sample(memory_B, 1)
    minibatch = [memory_B[-1]]
    print("minibatch: ", minibatch)
    mb_states_human, mb_actions_human, mb_rewards_human, mb_next_states, mb_dones = zip(*minibatch)
    mb_states_human = tf.convert_to_tensor(mb_states_human, dtype=tf.float32)
    mb_actions_human = tf.convert_to_tensor(mb_actions_human, dtype=tf.float32)
    mb_rewards_human = tf.convert_to_tensor(mb_rewards_human, dtype=tf.float32)
    return mb_states_human, mb_actions_human, mb_rewards_human
print("\n")
#print("memory_B: ", memory_B)
print(take_last(memory_B))