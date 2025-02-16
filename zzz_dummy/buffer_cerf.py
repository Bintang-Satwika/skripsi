import random
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.memory_B = []  # This will store tuples of (state, action, reward, next_state, done)
        
    def update_RL_memory(self, state, action, reward, next_state, done):
        """
        Save (s, a, r, s', done) to the replay buffer.
        
        For a multi-agent scenario:
          - state: np.array of shape (3, 14)
          - action: np.array of shape (3,)
          - reward: np.array of shape (3,)
          - next_state: np.array of shape (3, 14)
          - done: np.array of shape (3,)
        """
        self.memory_B.append((state, action, reward, next_state, done))
    
    # def take_RL_minibatch(self):
    #     """
    #     Sample a minibatch from the replay buffer.
    #     Each minibatch element contains multi-agent data:
    #       - state: (3, 14)
    #       - action: (3,)
    #       - reward: (3,)
    #       - next_state: (3, 14)
    #       - done: (3,)
    #     After stacking, the shapes become:
    #       - mb_states: (batch_size, 3, 14)
    #       - mb_actions: (batch_size, 3)
    #       - mb_rewards: (batch_size, 3)
    #       - mb_next_states: (batch_size, 3, 14)
    #       - mb_dones: (batch_size, 3)
    #     """
    #     minibatch = random.sample(self.memory_B, self.batch_size)
        
    #     # Unpack and stack the multi-agent data from each tuple
    #     mb_states = tf.convert_to_tensor(np.stack([data[0] for data in minibatch]), dtype=tf.float32)
    #     mb_actions = tf.convert_to_tensor(np.stack([data[1] for data in minibatch]), dtype=tf.float32)
    #     mb_rewards = tf.convert_to_tensor(np.stack([data[2] for data in minibatch]), dtype=tf.float32)
    #     mb_next_states = tf.convert_to_tensor(np.stack([data[3] for data in minibatch]), dtype=tf.float32)
    #     mb_dones = tf.convert_to_tensor(np.stack([data[4] for data in minibatch]), dtype=tf.float32)
        
    #     return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones
    def take_RL_minibatch(self):
        """Ambil minibatch dari buffer RL dengan mengambil 3 sample pertama."""
        # Ensure that there are at least 3 samples in the buffer
        if len(self.memory_B) < 100:
            raise ValueError("Not enough samples in the memory to form a minibatch.")
        
        # Instead of random.sample, we take the first 3 samples
        minibatch = self.memory_B[:100]
        
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = zip(*minibatch)
        mb_states = tf.convert_to_tensor(np.stack(mb_states), dtype=tf.float32)
        mb_actions = tf.convert_to_tensor(np.stack(mb_actions), dtype=tf.float32)
        mb_rewards = tf.convert_to_tensor(np.stack(mb_rewards), dtype=tf.float32)
        mb_next_states = tf.convert_to_tensor(np.stack(mb_next_states), dtype=tf.float32)
        mb_dones = tf.convert_to_tensor(np.stack(mb_dones), dtype=tf.float32)
        
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones

# Example usage:
if __name__ == '__main__':
    # Create a replay buffer with a minibatch size of 32.
    buffer = ReplayBuffer(batch_size=32)
    
    # Suppose you have a loop over episodes and timesteps.
    # Here, we simulate storing one experience per timestep for demonstration.
    for _ in range(100):
        # Dummy multi-agent data:
        state = np.random.randn(3, 14)        # shape: (3, 14)
        action = np.random.randint(0, 4, size=(3,))  # shape: (3,)
        reward = np.random.rand(3)              # shape: (3,)
        next_state = np.random.randn(3, 14)     # shape: (3, 14)
        done = np.random.choice([0, 1], size=(3,))    # shape: (3,)
        
        buffer.update_RL_memory(state, action, reward, next_state, done)
    
    # When ready to train, take a minibatch:
    mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = buffer.take_RL_minibatch()
    print("States batch shape:", mb_states.shape)
    print("Actions batch shape:", mb_actions.shape)
    print("Rewards batch shape:", mb_rewards.shape)
    print("Next States batch shape:", mb_next_states.shape)
    print("Dones batch shape:", mb_dones.shape)
