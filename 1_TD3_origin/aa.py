class TD3Agent:
    def __init__(self):
        # Initialize your agent here
        self.last_episode = None

    def main_loop(self):
        # Your main loop implementation
        for episode in range(1, 1001):  # Example loop
            # Your training code here
            self.last_episode = episode  # Set last_episode to the current episode
            print(f"Episode {episode} completed")

    def plot_results(self):
        # Your plot results implementation
        print("Plotting results...")

    def save_models(self, episode):
        # Your save models implementation
        print(f"Saving models for episode {episode}...")

if __name__ == "__main__":
    agent = TD3Agent()
    agent.main_loop()
    agent.plot_results()
    print("aaaa", agent.last_episode)
    agent.save_models(agent.last_episode)