import numpy as np
import random
'''
timestep pengerjaan pada workbench masih =0, seharusnya timestep berjalan sembari agent mengerjakan jobnya di workbench.
saat product sudah selesai, product belum menghilang dari conveyor namun sudah ada di completed product
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
        self.product_completed = []  # Buffer for finished products.

    def add_job(self, product_type):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        # Insert the job into section 0 if available; otherwise, add to the buffer.
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
        self.pending_reinsertion = False  # Flag: True if a job has been processed and is waiting to be reinserted.

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
        # Agent 1 fixed at index 3, Agent 2 fixed at index 7, Agent 3 fixed at index 11.
        self.fixed_positions = [3, 7, 11]
        # Define agent capabilities:
        # Agent 1: can process operations {1, 2}
        # Agent 2: can process operations {2, 3}
        # Agent 3: can process operations {1, 3}
        capabilities = [{1, 2}, {2, 3}, {1, 3}]
        self.window_size = 3  # (Not used in this processing loop but can be used for view display.)
        self.agents = []
        for i in range(3):
            agent = Agent(agent_id=i+1, capability=capabilities[i])
            agent.fixed_position = self.fixed_positions[i]
            agent.current_job = None
            agent.pending_reinsertion = False
            self.agents.append(agent)

    def step(self):
        print("\n--- New Timestep ---")
        # 1. Generate new jobs and move the conveyor.
        self.conveyor.generate_jobs()
        self.conveyor.move_conveyor()
        self.conveyor.display()

        # 2. For each agent, check the fixed position.
        # (The agent's fixed position is where it picks up a new job.)
        for agent in self.agents:
            fixed_index = agent.fixed_position
            job_at_fixed = self.conveyor.conveyor[fixed_index]
            if job_at_fixed is not None:
                if agent.current_job is None:
                    # Agent is idle.
                    required_ops = self.conveyor.job_details.get(job_at_fixed, [])
                    if required_ops:
                        required_op = required_ops[0]
                        if required_op in agent.capability:
                            # ACCEPT the job.
                            agent.current_job = job_at_fixed
                            # Remove the job from the conveyor.
                            self.conveyor.conveyor[fixed_index] = None
                            print(f"Agent {agent.agent_id} ACCEPTS job {job_at_fixed} at fixed position {fixed_index} (requires op {required_op}).")
                        else:
                            # DECLINE the job.
                            print(f"Agent {agent.agent_id} DECLINES job {job_at_fixed} at fixed position {fixed_index} (requires op {required_op} not in {agent.capability}).")
                    else:
                        print(f"Agent {agent.agent_id} found job {job_at_fixed} with no operation details.")
                else:
                    # Agent is busy, so ignore new job at fixed position.
                    print(f"Agent {agent.agent_id} is busy and ignores job {job_at_fixed} at fixed position {fixed_index}.")
            else:
                print(f"No job at Agent {agent.agent_id}'s fixed position {fixed_index}.")

        # 3. Process one operation of each agent's current job.
        for agent in self.agents:
            if agent.current_job is not None:
                # If the job has not yet been processed in a previous timestep waiting for reinsert,
                # process one operation.
                if not agent.pending_reinsertion:
                    completed_op = agent.process_job(self.conveyor.job_details)
                    if completed_op is not None:
                        print(f"Agent {agent.agent_id} processes operation {completed_op} of job {agent.current_job}.")
                        agent.pending_reinsertion = True  # Mark that the job is ready for reinsertion.
                    else:
                        print(f"Agent {agent.agent_id} attempted to process job {agent.current_job}, but no operations were found.")
                # Now, attempt to reinsert the job at the agent's fixed position.
                fixed_index = agent.fixed_position
                if agent.pending_reinsertion:
                    if self.conveyor.conveyor[fixed_index] is None:
                        # Reinsert the job at the fixed position.
                        self.conveyor.conveyor[fixed_index] = agent.current_job
                        print(f"Job {agent.current_job} reinserted into conveyor at fixed position {fixed_index} with remaining ops {self.conveyor.job_details[agent.current_job]}.")
                        # Check if the job is now complete.
                        if not self.conveyor.job_details[agent.current_job]:
                            print(f"Job {agent.current_job} is fully completed by Agent {agent.agent_id}. Moving to product_completed.")
                            self.conveyor.product_completed.append(agent.current_job)
                            del self.conveyor.job_details[agent.current_job]
                        # Clear the workbench and reset the flag.
                        agent.current_job = None
                        agent.pending_reinsertion = False
                    else:
                        # The fixed position is occupied; the agent must wait.
                        print(f"Agent {agent.agent_id} waits: fixed position {fixed_index} is occupied. Job {agent.current_job} remains on workbench.")
            else:
                print(f"Agent {agent.agent_id} is idle (no job on workbench).")

        self.display_state()

    def display_state(self):
        print("\nCurrent System State:")
        self.conveyor.display()
        for agent in self.agents:
            status = f"Processing job {agent.current_job}" if agent.current_job is not None else "Idle"
            # For display purposes, show the window as the 'window_size' sections preceding the fixed position.
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
    for timestep in range(20):
        print(f"\nTime Step {timestep + 1}")
        fj_system.step()

    print("\nTotal Jobs Generated:", fj_system.conveyor.total_jobs)
    print("Buffer:", fj_system.conveyor.buffer_jobs)
    print("Completed Products:", fj_system.conveyor.product_completed)
