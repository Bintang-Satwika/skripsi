import os
import tensorflow as tf
from tensorflow.keras import layers, models

class TD3Loader:
    def __init__(self, state_dim=8, action_dim=2, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir

        self.actor = self.create_actor_network()
        self.critic_1 = self.create_critic_network()
        self.critic_2 = self.create_critic_network()

        self.target_actor = self.create_actor_network()

        self.target_critic_1 = self.create_critic_network()

        self.target_critic_2 = self.create_critic_network()

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
    
    def load_models(self, episode):
        """Memuat bobot model Actor dan Critic dari file .h5."""
        actor_path = os.path.join(self.load_dir, f'actor_episode_{episode}.h5')
        target_actor_path = os.path.join(self.load_dir, f'target_actor_episode_{episode}.h5')
        critic_1_path = os.path.join(self.load_dir, f'critic_1_episode_{episode}.h5')
        target_critic_1_path = os.path.join(self.load_dir, f'target_critic_1_episode_{episode}.h5')
        critic_2_path = os.path.join(self.load_dir, f'critic_2_episode_{episode}.h5')
        target_critic_2_path = os.path.join(self.load_dir, f'target_critic_2_episode_{episode}.h5')

        for path in [actor_path, target_actor_path, critic_1_path, target_critic_1_path, critic_2_path, target_critic_2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.actor.load_weights(actor_path)
        self.target_actor.load_weights(target_actor_path)
        self.critic_1.load_weights(critic_1_path)
        self.target_critic_1.load_weights(target_critic_1_path)
        self.critic_2.load_weights(critic_2_path)
        self.target_critic_2.load_weights(target_critic_2_path)

        print(f'Models loaded from episode {episode}')

        return self.actor, self.target_actor, self.critic_1, self.target_critic_1, self.critic_2, self.target_critic_2


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.join(current_dir, 'ruled_based_5')
    loader = TD3Loader(load_dir=load_dir)
    actor, target_actor, critic_1, target_critic_1, critic_2, target_critic_2 = loader.load_models(episode=140)
    bias_output = actor.layers[-1].get_weights()[-1]
    print("bias_output:", bias_output)