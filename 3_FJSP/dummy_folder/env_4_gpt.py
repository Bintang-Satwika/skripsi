import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces

# ============================================================================
# CircularConveyor: Manages job arrivals, conveyor movement, buffering,
# and keeps track of completed products.
# ============================================================================
class CircularConveyor:
    def __init__(self, num_sections: int, max_capacity: float, arrival_rate: float, num_agents: int, n_jobs: int):
        self.num_sections = num_sections            # Total number of sections
        self.max_capacity = max_capacity            # Maximum fill percentage (e.g., 75%)
        self.arrival_rate = arrival_rate            # Poisson arrival rate for jobs
        self.conveyor = [None] * num_sections       # Initialize empty conveyor
        self.buffer_jobs = []                       # Buffer for jobs when the entry is full
        self.total_jobs = {"A": 0, "B": 0, "C": 0}    # Job counters per product
        # Define the operation sequences for each product:
        #   Product A: operations 1, 2, 3
        #   Product B: operations 2, 3
        #   Product C: operations 1, 2
        self.product_operations = {
            "A": [1, 2, 3],
            "B": [2, 3],
            "C": [1, 2]
        }
        self.job_details = {}  # For each job, stores its remaining operations.
        self.product_completed = []  # Buffer for finished products.
        self.n_jobs = n_jobs
        self.sum_n_jobs = 0
        self.num_agents = num_agents

    def add_job(self, product_type: str):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        # Insert the job into section 0 if available and if there are enough free slots beyond agents.
        if (sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None) > self.num_agents):
            self.conveyor[0] = job_label
        else:
            self.buffer_jobs.append(job_label)

    def move_conveyor(self):
        """Moves jobs forward in a circular manner."""
        last_job = self.conveyor[-1]
        for i in range(self.num_sections - 1, 0, -1):
            self.conveyor[i] = self.conveyor[i - 1]
        self.conveyor[0] = last_job
        # If the first section is empty and capacity allows, load a job from the buffer.
        if (self.buffer_jobs and 
            sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None) > self.num_agents):
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates new jobs based on a Poisson process (if below maximum n_jobs)."""
        if self.sum_n_jobs < self.n_jobs:
            new_job = min(1, np.random.poisson(self.arrival_rate, size=1)[0])
            if new_job == 1:
                self.sum_n_jobs += 1
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
    def __init__(self, agent_id: int, position: int, operation_capability: list, speed: float, base_processing_time: float, window_size: int, num_agent: int):
        # State attributes
        self.position = position  # Fixed position on the conveyor (0-indexed)
        self.operation_capability = np.array(operation_capability)  # e.g. [1,2]
        self.operation_now = 0   # The next operation (if any)
        self.status_all = [0] * num_agent  # e.g. [0,0,0] (for multi-agent status indicators)
        self.remaining_operation = np.zeros(window_size)  # Window observation (e.g., remaining operations count)
        # Fixed properties
        self.id = agent_id
        self.speed = speed
        self.base_processing_time = base_processing_time
        self.current_job = None   # The job on the workbench
        self.processing_time_remaining = 0
        self.pending_reinsertion = False
        self.workbench = {}  # Dictionary to store the job (key: job label, value: remaining operations)

    def build_state(self):
        """
        Build the state representation as a numpy array.
        State vector: [position, operation_capability (2 elements), operation_now, status_all (num_agent elements), remaining_operation (window_size elements)]
        Total dimension = 1 + 2 + 1 + num_agent + window_size.
        """
        return np.hstack([
            np.array([self.position]),
            np.array(self.operation_capability),
            np.array([self.operation_now]),
            np.array(self.status_all),
            np.array(self.remaining_operation)
        ])

    def processing_time(self):
        """Compute processing time (per operation) based on base_processing_time and speed."""
        return np.ceil(self.base_processing_time / self.speed)

    def start_job(self):
        """Initialize processing time when a job is accepted."""
        self.processing_time_remaining = int(math.ceil(self.base_processing_time / self.speed))

    def process(self, job_details):
        """
        Decrement the processing time. If processing time reaches zero, complete one operation.
        Returns the completed operation (or None if not complete).
        """
        if self.current_job is not None:
            if self.processing_time_remaining > 0:
                self.processing_time_remaining -= 1
                return None
            if self.processing_time_remaining == 0:
                # Complete one operation from the job
                if self.current_job in job_details and len(job_details[self.current_job]) > 0:
                    return job_details[self.current_job].pop(0)
        return None

# ============================================================================
# FJSPEnv: Gymnasium Environment for the Flexible Job Shop Problem.
# ============================================================================
class FJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, window_size: int, num_agents: int, max_steps: int):
        super(FJSPEnv, self).__init__()
        self.window_size = window_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0

        # Conveyor parameters
        self.num_sections = 12
        self.max_capacity = 0.75
        self.arrival_rate = 0.4
        self.n_jobs = 20

        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, num_agents, n_jobs=self.n_jobs)

        # Agent configuration
        # Fixed positions (0-indexed): e.g., Agent1: 3, Agent2: 7, Agent3: 11
        self.agent_positions = [3, 7, 11]
        self.agent_operation_capability = [[1, 2], [2, 3], [1, 3]]
        self.agent_speeds = [1, 2, 3]  # (for example, Agent2 is 2x faster; Agent3 is 3x faster)
        self.base_processing_times = [6, 4, 2]  # Base processing time for each agent
        # For state, we assume status is stored in indices  (we set status_all later)
        self.agent_status_location = 3  # (Assume first element of status_all is at index 3)
        self.agent_many_operations = 2  # length of operation_capability
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i + 1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                base_processing_time=self.base_processing_times[i],
                window_size=self.window_size,
                num_agent=self.num_agents
            )
            self.agents.append(agent)

        # Observation space: each agent’s state vector has dimension = 1 + 2 + 1 + num_agents + window_size.
        state_dim = 1 + 2 + 1 + self.num_agents + self.window_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents, state_dim), dtype=np.float32)
        # Action space: For each agent, 4 discrete actions (0: ACCEPT, 1: WAIT, 2: DECLINE, 3: CONTINUE).
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)

        self.observation_all = None

    def _get_obs(self):
        obs = []
        for agent in self.agents:
            obs.append(agent.build_state())
        self.observation_all = np.array(obs)
        return self.observation_all

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.conveyor = CircularConveyor(self.num_sections, self.max_capacity, self.arrival_rate, self.num_agents, n_jobs=self.n_jobs)
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i + 1,
                position=self.agent_positions[i],
                operation_capability=self.agent_operation_capability[i],
                speed=self.agent_speeds[i],
                base_processing_time=self.base_processing_times[i],
                window_size=self.window_size,
                num_agent=self.num_agents
            )
            self.agents.append(agent)
        return self._get_obs(), {}

    def update_state(self, observation_all, actions_all):
        next_observation_all = []
        # For simplicity, we assume that the agent’s status is stored at index = self.agent_status_location in its state vector.
        for i, agent in enumerate(self.agents):
            print(f"\nAgent {i+1}:")
            observation = observation_all[i].copy()
            yr = agent.position
            status_location = self.agent_status_location  # e.g., index 3 in state vector
            # The agent’s window is defined as the sections [yr-1, yr-2, ..., yr-window_size]
            window_sections = [yr - r for r in range(1, self.window_size + 1)]
            window_agent_product = np.array(self.conveyor.conveyor)[window_sections]
            print("Window sections:", window_sections, "->", window_agent_product)
            print("Agent workbench:", agent.workbench)

            # --- Process actions ---
            if actions_all[i] == 0:  # ACCEPT
                # Two cases: if job is already on workbench or job is on conveyor at fixed position.
                if agent.workbench and observation[status_location] == 1:
                    print("ACCEPT already on workbench.")
                    observation[status_location] = 2  # Change status from accept to working.
                    list_operation = list(agent.workbench.values())[0]  # Get the remaining operations.
                    print("Remaining operations in workbench:", list_operation)
                    if list_operation[0] in agent.operation_capability:
                        # Update operation_now in observation (stored at index 1+agent_many_operations)
                        select_operation = agent.operation_capability[0]  # For simplicity, choose first capability.
                        observation[1 + self.agent_many_operations] = select_operation
                    else:
                        print("FAILED ACTION: operation capability is False.")
                elif not agent.workbench:
                    req_ops = self.conveyor.job_details.get(self.conveyor.conveyor[yr], [])
                    if self.conveyor.conveyor[yr] is not None and req_ops and req_ops[0] in agent.operation_capability and observation[status_location] == 0:
                        print("ACCEPT from conveyor at fixed position.")
                        observation[status_location] = 1  # idle becomes accept.
                        # Move the job to workbench.
                        agent.workbench[self.conveyor.conveyor[yr]] = req_ops
                        self.conveyor.conveyor[yr] = None
                        observation[1 + self.agent_many_operations] = 0  # reset operation_now
                    else:
                        print("FAILED ACTION: cannot accept job from conveyor.")
                else:
                    print("FAILED ACTION: workbench is not empty.")

            elif actions_all[i] == 1:  # WAIT
                print("WAIT: Agent", i+1, "waits for job arrival in its window.")
                # For WAIT, do nothing special – the agent remains idle (no job taken).
                # You might update remaining_operation in the observation later.
            elif actions_all[i] == 2:  # DECLINE
                # Remove the job at the fixed position if exists.
                if self.conveyor.conveyor[yr] is not None:
                    print(f"DECLINE: Agent {i+1} declines job {self.conveyor.conveyor[yr]} at fixed position {yr}.")
                    self.conveyor.conveyor[yr] = None
                    observation[status_location] = 0  # Reset status to idle.
                else:
                    print(f"DECLINE: No job at fixed position for Agent {i+1}.")
            elif actions_all[i] == 3:  # CONTINUE
                print("CONTINUE: Agent", i+1, "continues its current action.")
                # Do nothing – agent continues processing if busy, or remains idle.
            else:
                print("Unknown action.")

            next_observation_all.append(observation)
        next_observation_all = np.array(next_observation_all)

        # After processing actions, update the window information in the observation.
        # We call move_conveyor() and generate_jobs() to simulate conveyor dynamics.
        self.conveyor.move_conveyor()
        self.conveyor.generate_jobs()
        for i, agent in enumerate(self.agents):
            yr = agent.position
            window_sections = [yr - r for r in range(1, self.window_size + 1)]
            job_details_value = [self.conveyor.job_details.get(self.conveyor.conveyor[j], []) for j in window_sections]
            # Update the last 'window_size' elements in the observation with the number of remaining operations.
            for j, value in enumerate(job_details_value):
                next_observation_all[i, -self.window_size + j] = len(value)
        return next_observation_all

    def step(self, actions):
        """
        actions: array of length num_agents, each in {0: ACCEPT, 1: WAIT, 2: DECLINE, 3: CONTINUE}
        """
        self.step_count += 1

        # (1) Generate new jobs and move the conveyor.
        self.conveyor.generate_jobs()
        self.conveyor.move_conveyor()

        # (2) Transfer a job at the designated transfer point (if any) [if needed].
        # (In this code, we do not use a designated transfer point because each agent picks up from its fixed position.)

        # (3) Let each agent process its current job (if any).
        for agent in self.agents:
            if agent.current_job is not None:
                op_completed = agent.process(self.conveyor.job_details)
                if op_completed is not None:
                    print(f"Agent {agent.id} completes operation {op_completed} of job {agent.current_job}.")
                    # If job still has remaining operations, try to reinsert it at fixed position.
                    fixed_index = agent.position
                    if self.conveyor.conveyor[fixed_index] is None:
                        self.conveyor.conveyor[fixed_index] = agent.current_job
                        print(f"Job {agent.current_job} reinserted at fixed position {fixed_index} with remaining ops {self.conveyor.job_details[agent.current_job]}.")
                        if len(self.conveyor.job_details[agent.current_job]) == 0:
                            print(f"Job {agent.current_job} is fully completed. Moving to product_completed.")
                            self.conveyor.product_completed.append(agent.current_job)
                            del self.conveyor.job_details[agent.current_job]
                            self.conveyor.conveyor[fixed_index] = None
                        agent.current_job = None
                        agent.processing_time_remaining = 0
                    else:
                        print(f"Agent {agent.id} waits: fixed position {fixed_index} is occupied.")
        # (4) Update state based on actions.
        next_obs = self.update_state(self.observation_all, actions)
        self.observation_all = next_obs

        # Simple reward: negative of the buffer size.
        reward = -len(self.conveyor.buffer_jobs)
        done = self.step_count >= self.max_steps
        truncated = False
        info = {"actions": actions}
        return next_obs, reward, done, truncated, info

    def render(self, mode=None):
        #print(f"Time Step: {self.step_count}")
        self.conveyor.display()
        for i, agent in enumerate(self.agents):
            status = agent.current_job if agent.current_job is not None else "Idle"
           # print(f"Agent {agent.id} at position {agent.position}: {status}")
      #  print("-" * 50)

    def reset_state(self):
        return self._get_obs()

# ============================================================================
# Main: Run the environment using Gymnasium format.
# ============================================================================
if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=200)
    obs, info = env.reset(seed=42)
    env.observation_all = obs  # Set initial observation
    env.render()
    total_reward = 0
    done = False
    truncated = False
    print("Initial state:\n", obs)
    while not done and not truncated:
        print("\nStep:", env.step_count)
        # For testing, here we choose a fixed action, e.g., all agents ACCEPT (action 0)
        actions = env.action_space.sample()  # or set actions = [0, 1, 2, ...] as needed
        # For demonstration, we force a particular action set. For example, [0, 1, 3]
        print("Actions chosen:", actions)
        next_state, reward, done, truncated, info = env.step(actions)
        print("Reward:", reward)
        total_reward += reward
        env.render()
        print("-" * 100)
        obs = next_state
    print("Episode complete. Total Reward:", total_reward)
