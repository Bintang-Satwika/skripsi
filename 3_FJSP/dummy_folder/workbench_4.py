import numpy as np
import random
import math
from circular_conveyor_2 import CircularConveyor

''''
sudah memperbaiki agar maksimal job dari poisson distribusion adalah 20, jika lebih dari itu maka tidak ada job baru
'''

class Agent:
    def __init__(self, agent_id, position, operation_capability, speed, base_processing_time, window_size):

        # Agent state attributes
        self.position = position
        self.operation_capability = np.array(operation_capability)  # Ensure it's a numpy array
        self.operation_now = 0
        self.status = 0
        self.remaining_operation = np.zeros(window_size)  # Use numpy array for efficiency

        # Agent fixed properties
        self.id = agent_id
        self.speed = speed
        self.base_processing_time = base_processing_time
        self.current_job = None
        self.processing_time_remaining = 0
        self.pending_reinsertion = False

    def build_state(self):
        """
        Build the state representation for the agent as a numpy array.
        """
        return np.hstack([
            np.array([self.position]),  # Position as an array
            self.operation_capability,  # Already a numpy array
            np.array([self.operation_now, self.status]),  # Convert to array
            self.remaining_operation  # Already a numpy array
        ])

    def processing_time(self):
        """
        Compute and return processing time as a numpy array.
        """
        return np.array(np.ceil(self.base_processing_time / self.speed))
    


class FlexibleJobShopSystem:
    def __init__(self, num_sections: int, max_capacity: float, arrival_rate: float, num_agents: int, n_jobs: int):
        self.window_size = 3  # Sections the agent can observe
        self.num_sections = num_sections
        self.max_capacity = max_capacity
        self.arrival_rate = arrival_rate
        self.num_agents = num_agents
        self.n_jobs = n_jobs

        # Agent configurations
        self.agent_position_all = np.array([3, 7, 11])
        self.agent_operation_capability_all = [np.array([1, 2]), np.array([2, 3]), np.array([1, 3])]
        self.agent_velocity_all = np.array([1, 2, 0.5])
        self.base_processing_time_all = np.array([6, 4, 2])  # [Op1, Op2, Op3]

        self.conveyor = CircularConveyor(num_sections, max_capacity, arrival_rate, num_agents=3, n_jobs=20)

        # Initialize agent states and processing times as numpy arrays
        self.agents_states = []
        self.agents_process_time = []

        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i + 1,
                position=self.agent_position_all[i],
                operation_capability=self.agent_operation_capability_all[i],
                speed=self.agent_velocity_all[i],
                base_processing_time=self.base_processing_time_all[i],
                window_size=self.window_size
            )

            self.agents_states.append(agent.build_state())
            self.agents_process_time.append(agent.processing_time())

        # Convert lists to numpy arrays for better computation
        self.agents_states = np.vstack(self.agents_states)
        self.agents_process_time = np.hstack(self.agents_process_time)

        print("self.agent_states:\n", self.agents_states)
        print("self.agents_process_time:\n", self.agents_process_time)

    def reward_calculation(self):
        """
        Calculate the reward based on the current state of the system.
        """
        return
    
    def FAA_action(self):
        return
    
    def step(self):
        print("\n--- New Timestep ---")
        # 1. Generate new jobs and move the conveyor.
        self.conveyor.generate_jobs()
        self.conveyor.move_conveyor()

        self.conveyor.display()
        for r in range(0, self.num_agents):
            print(f"\nAgent {r + 1}")
            state= self.agents_states[r]
            
            print("state:\n", state)
            process_time = self.agents_process_time[r]
            velocity = self.agent_velocity_all[r]
            position = self.agent_position_all[r]
            start = max(0, position - self.window_size)
            window_indices = list(range(start, position))
            print("window_indices:", window_indices)
            window_jobs = [self.conveyor.conveyor[idx] if self.conveyor.conveyor[idx] is not None else None for idx in window_indices]
            print(f"Window: {window_jobs}")





# ============================================================================
# Example Simulation
# ============================================================================
if __name__ == "__main__":
    # Create the Flexible Job Shop System.
    fj_system = FlexibleJobShopSystem(num_sections=12, max_capacity=0.75, arrival_rate=0.3, num_agents=3, n_jobs=20)

    # # Run the simulation for 20 timesteps.
    for timestep in range(1):
         print(f"\nTime Step {timestep + 1}")
         fj_system.step()
         if len(fj_system.conveyor.product_completed)>=20:
            break

    # print("\nTotal Jobs Generated:", fj_system.conveyor.total_jobs)
    # print("Buffer:", fj_system.conveyor.buffer_jobs)
    # print("Completed Products:", fj_system.conveyor.product_completed)
    # print("len(fj_system.conveyor.product_completed):", len(fj_system.conveyor.product_completed))
