import os
import tensorflow as tf
from tensorflow.keras import layers, models

class DDQNLoader:
    def __init__(self, state_dim=50, action_dim=5, load_dir='file_path'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.load_dir = load_dir

        self.dqn_network = self.create_dqn_network()
        self.target_dqn_network = self.create_dqn_network()

    def create_dqn_network(self):
        """Membuat model Q-network dengan layer fully connected."""
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        output = layers.Dense(self.action_dim, activation='linear')(x)
        model = models.Model(inputs=state_input, outputs=output)
        return model

    def load_models(self, episode):
        """Memuat bobot model DQN dan target DQN dari file .h5."""
        dqn_load_path = os.path.join(self.load_dir, f'dqn_episode_{episode}.h5')
        target_dqn_load_path = os.path.join(self.load_dir, f'target_dqn_episode_{episode}.h5')

        for path in [dqn_load_path, target_dqn_load_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.dqn_network.load_weights(dqn_load_path)
        self.target_dqn_network.load_weights(target_dqn_load_path)
        print(f'Models loaded from episode {episode}')

        return self.dqn_network, self.target_dqn_network


if __name__ == "__main__":
    loader = DDQNLoader(state_dim=50, action_dim=5,load_dir='D:\\KULIAH\\skripsi\\CODE\\skripsi\\double_dqn\\highwaymodel_1')
    dqn_net, target_dqn_net = loader.load_models(episode=80)
    bias_output = dqn_net.layers[-1].get_weights()[-1]
    print("bias_output:", bias_output)