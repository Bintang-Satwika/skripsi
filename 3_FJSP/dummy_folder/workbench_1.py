import numpy as np
import random
'''
 agent masih salah dalam mengembalikan product dari workbenc ke conveyor. 
 karena agent masih mengembalikan product pada conveyor[0] yang seharusnya mengembalikan pada conveyor[yr]
 '''
# ============================================================================
# CircularConveyor: Manages job arrivals, conveyor movement, buffering,
# and keeps track of completed products.
# ============================================================================
class CircularConveyor:
    def __init__(self, num_sections, max_capacity, arrival_rate, num_agents):
        self.num_sections = num_sections            # Total number of sections
        self.max_capacity = max_capacity            # Maximum fill percentage (e.g., 75%)
        self.arrival_rate = arrival_rate            # Poisson arrival rate for jobs
        self.conveyor = [None] * num_sections       # Initialize empty conveyor
        self.buffer_jobs = []                       # Buffer for jobs when the entry is full
        self.total_jobs = {"A": 0, "B": 0, "C": 0}    # Job counters per product
        # Define the operation sequences for each product:
        #   - Product A: operations 1, 2, 3
        #   - Product B: operations 2, 3
        #   - Product C: operations 1, 2
        self.product_operations = {
            "A": [1, 2, 3],
            "B": [2, 3],
            "C": [1, 2]
        }
        self.job_details = {}  # For each job, stores its remaining operations.
        self.product_completed = []  # Buffer to hold finished products.

    def add_job(self, product_type):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job's sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        # Insert the job into the first section if available; otherwise, add to buffer.
        if self.conveyor[0] is None:
            self.conveyor[0] = job_label
        else:
            self.buffer_jobs.append(job_label)

    def move_conveyor(self):
        """Moves jobs forward in a circular manner."""
        last_job = self.conveyor[-1]
        for i in range(self.num_sections - 1, 0, -1):
            self.conveyor[i] = self.conveyor[i - 1]
        self.conveyor[0] = last_job
        # If the first section is empty, load a job from the buffer.
        if self.buffer_jobs and self.conveyor[0] is None:
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates new jobs based on a Poisson process (if below maximum capacity)."""
        # np.random.seed(10)  # DO NOT DELETE THIS LINE if reproducibility is required.
        num_new_jobs = np.random.poisson(self.arrival_rate)
        for _ in range(num_new_jobs):
            if sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections:
                product_type = random.choice(list(self.total_jobs.keys()))
                self.add_job(product_type)

    def display(self):
        """Displays the state of the conveyor, the buffer, and completed products."""
        conveyor_state = " <-> ".join([str(j) if j is not None else "---" for j in self.conveyor])
        print("Conveyor:", conveyor_state)
        print("Buffer:", self.buffer_jobs)
        print("Completed Products:", self.product_completed)

# ============================================================================
# Agent: Represents a processing unit (robot) with a workbench.
# ============================================================================
class Agent:
    def __init__(self, agent_id, capability):
        self.agent_id = agent_id
        self.capability = capability  # A set of operations the agent can process.
        self.fixed_position = None    # To be set externally.
        self.current_job = None       # The job currently on the workbench.

    def process_job(self, job_details):
        """
        Processes one processing step (one operation) of the job on the workbench.
        Returns the completed operation (or None if no job is present).
        """
        if self.current_job and job_details.get(self.current_job, []):
            # Process (remove) the first operation.
            return job_details[self.current_job].pop(0)
        return None

# ============================================================================
# FlexibleJobShopSystem: Contains the conveyor and 3 agents (each with one workbench).
# ============================================================================
class FlexibleJobShopSystem:
    def __init__(self, num_sections=12, max_capacity=0.75, arrival_rate=0.3):
        # Create the circular conveyor.
        self.conveyor = CircularConveyor(num_sections, max_capacity, arrival_rate, num_agents=3)
        # Set fixed positions for agents (zero-indexed):
        # Agent 1 is located at index 3,
        # Agent 2 is located at index 7,
        # Agent 3 is located at index 11.
        self.fixed_positions = [3, 7, 11]
        # Define agent capabilities:
        # Agent 1: can process operations {1, 2}
        # Agent 2: can process operations {2, 3}
        # Agent 3: can process operations {1, 3}
        capabilities = [{1, 2}, {2, 3}, {1, 3}]
        self.window_size = 3  # Each agent sees the fixed position and the previous (window_size) sections.
        self.agents = []
        for i in range(3):
            agent = Agent(agent_id=i+1, capability=capabilities[i])
            agent.fixed_position = self.fixed_positions[i]
            agent.current_job = None
            self.agents.append(agent)

    def step(self):
        print("\n--- New Timestep ---")
        # 1. Generate new jobs and move the conveyor.
        self.conveyor.generate_jobs()
        self.conveyor.move_conveyor()
        self.conveyor.display()

        # 2. For each agent, check the fixed position.
        # The agent's "view" (window) is defined as the block of 'window_size' sections immediately preceding its fixed position.
        # For example, for an agent fixed at index 3, its window covers indices 0, 1, 2.
        for agent in self.agents:
            fixed_index = agent.fixed_position  # The agent's fixed position.
            job_at_fixed = self.conveyor.conveyor[fixed_index]
            if job_at_fixed is not None:
                if agent.current_job is None:
                    # Agent is idle. Check the next required operation of the job.
                    required_ops = self.conveyor.job_details.get(job_at_fixed, [])
                    if required_ops:
                        required_op = required_ops[0]
                        if required_op in agent.capability:
                            # ACCEPT the job: remove it from the conveyor and place on workbench.
                            agent.current_job = job_at_fixed
                            self.conveyor.conveyor[fixed_index] = None
                            print(f"Agent {agent.agent_id} ACCEPTS job {job_at_fixed} at fixed position {fixed_index} (requires op {required_op}).")
                        else:
                            # DECLINE the job (job remains on conveyor).
                            print(f"Agent {agent.agent_id} DECLINES job {job_at_fixed} at fixed position {fixed_index} (requires op {required_op} not in {agent.capability}).")
                    else:
                        print(f"Agent {agent.agent_id} found job {job_at_fixed} with no operation details.")
                else:
                    # Agent is busy; decline any job at its fixed position.
                    print(f"Agent {agent.agent_id} is busy and DECLINES job {job_at_fixed} at fixed position {fixed_index}.")
            else:
                print(f"No job at Agent {agent.agent_id}'s fixed position {fixed_index}.")

        # 3. Process one operation of each agent's job.
        for agent in self.agents:
            if agent.current_job is not None:
                completed_op = agent.process_job(self.conveyor.job_details)
                if completed_op is not None:
                    print(f"Agent {agent.agent_id} processes operation {completed_op} of job {agent.current_job}.")
                    # Check if there are more operations.
                    if self.conveyor.job_details[agent.current_job]:
                        # Job still has remaining operations: reinsert it into the conveyor.
                        if self.conveyor.conveyor[0] is None:
                            self.conveyor.conveyor[0] = agent.current_job
                            print(f"Job {agent.current_job} reinserted into conveyor at section 0 with remaining ops {self.conveyor.job_details[agent.current_job]}.")
                        else:
                            self.conveyor.buffer_jobs.append(agent.current_job)
                            print(f"Job {agent.current_job} added to buffer with remaining ops {self.conveyor.job_details[agent.current_job]}.")
                    else:
                        # All operations have been completed.
                        print(f"Job {agent.current_job} is fully completed by Agent {agent.agent_id}. Moving product to product_completed.")
                        self.conveyor.product_completed.append(agent.current_job)
                        # Optionally, remove its details.
                        del self.conveyor.job_details[agent.current_job]
                else:
                    print(f"Agent {agent.agent_id} attempted to process job {agent.current_job}, but no operations found.")
                # Clear the workbench after processing.
                agent.current_job = None
            else:
                print(f"Agent {agent.agent_id} is idle (no job on workbench).")

        self.display_state()

    def display_state(self):
        print("\nCurrent System State:")
        self.conveyor.display()
        for agent in self.agents:
            status = f"Processing job {agent.current_job}" if agent.current_job is not None else "Idle"
            # Each agent's window covers the 'window_size' sections immediately preceding its fixed position.
            start = max(0, agent.fixed_position - self.window_size)
            window_indices = list(range(start, agent.fixed_position))
            window_jobs = [self.conveyor.conveyor[idx] if self.conveyor.conveyor[idx] is not None else "---" for idx in window_indices]
            print(f"Agent {agent.agent_id} at fixed position {agent.fixed_position}: {status} | Window: {window_jobs}")
        print("-" * 50)

# ============================================================================
# Example Simulation
# ============================================================================
if __name__ == "__main__":
    # Create the Flexible Job Shop System.
    fj_system = FlexibleJobShopSystem(num_sections=12, max_capacity=0.75, arrival_rate=0.3)
    # Run the simulation for 10 timesteps.
    for timestep in range(10):
        print(f"\nTime Step {timestep + 1}")
        fj_system.step()

    print("\nTotal Jobs Generated:", fj_system.conveyor.total_jobs)
    print("Buffer:", fj_system.conveyor.buffer_jobs)
    print("Completed Products:", fj_system.conveyor.product_completed)
