import numpy as np

class AgentState:
    def __init__(self, agent_id, position, operations, window_size):
        """
        Initialize the agent's state.
        :param agent_id: Unique ID for the agent.
        :param position: The position of the agent (y_r).
        :param operations: A list of operations the agent can perform (O_r).
        :param window_size: The size of the observation window.
        """
        self.agent_id = agent_id
        self.position = position
        self.operations = operations
        self.window_size = window_size
        self.current_operation = 0  # Default: idle
        self.status = 0  # Default: idle (0: idle, 1: accepting, 2: working, 3: completed)
        self.remaining_operations = np.zeros(window_size)  # Jobs in view

    def update_state(self, current_operation, status, remaining_ops):
        """
        Update the agent's state.
        :param current_operation: The current operation type (O_t^r).
        :param status: The status of the agent.
        :param remaining_ops: Remaining operations within the agent's observation window.
        """
        self.current_operation = current_operation
        self.status = status
        self.remaining_operations = np.array(remaining_ops)

    def get_local_observation(self):
        """
        Generate the local observation vector for the agent.
        :return: A dictionary of the agent's observation components.
        """
        return {
            "Position": self.position,
            "Operations": self.operations,
            "Current Operation": self.current_operation,
            "Remaining Operations": self.remaining_operations.tolist(),
        }


class Environment:
    def __init__(self, num_agents, window_size):
        """
        Initialize the environment.
        :param num_agents: Total number of agents in the environment.
        :param window_size: Size of the observation window for each agent.
        """
        self.num_agents = num_agents
        self.window_size = window_size
        self.agents = []
        self.global_status = np.zeros(num_agents)  # Z_t: Status of all agents

    def add_agent(self, agent):
        """
        Add an agent to the environment.
        :param agent: An instance of AgentState.
        """
        self.agents.append(agent)

    def update_global_status(self):
        """
        Update the global status vector Z_t based on the statuses of all agents.
        """
        self.global_status = np.array([agent.status for agent in self.agents])

    def get_global_observation(self):
        """
        Generate the global observation for all agents.
        :return: Global observation including all agents' statuses and individual observations.
        """
        self.update_global_status()
        observations = []
        for agent in self.agents:
            local_obs = agent.get_local_observation()
            local_obs["Global Status"] = self.global_status.tolist()
            observations.append(local_obs)
        return observations


# Example Usage
if __name__ == "__main__":
    window_size = 3
    env = Environment(num_agents=3, window_size=window_size)

    # Initialize agents
    agent1 = AgentState(agent_id=1, position=1, operations=[1, 2], window_size=window_size)
    agent2 = AgentState(agent_id=2, position=2, operations=[2, 3], window_size=window_size)
    agent3 = AgentState(agent_id=3, position=3, operations=[1, 3], window_size=window_size)

    # Add agents to the environment
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)

    # Update agent states
    agent1.update_state(current_operation=1, status=2, remaining_ops=[3, 2, 1])
    agent2.update_state(current_operation=2, status=1, remaining_ops=[2, 3, 0])
    agent3.update_state(current_operation=0, status=0, remaining_ops=[1, 0, 0])

    # Get global observation
    global_observation = env.get_global_observation()
    for obs in global_observation:
        print(obs)
