import numpy as np
import random

class CircularConveyor:
    def __init__(self, num_sections=12, max_capacity=0.75, arrival_rate=0.03):
        self.num_sections = num_sections              # Number of conveyor sections
        self.max_capacity = max_capacity              # Maximum fill percentage
        self.arrival_rate = arrival_rate              # Poisson job arrival rate
        self.conveyor = [None] * num_sections         # Initialize empty conveyor
        self.buffer_jobs = []                         # Buffer for excess jobs
        self.total_jobs = {"A": 0, "B": 0, "C": 0, "D": 0}  # Job counters for each product type
        self.product_operations = {
            "A": [1, 2, 3],  # Operations for Product A
            "B": [2, 3],     # Operations for Product B
            "C": [1, 2],     # Operations for Product C
            "D": [3]         # Operations for Product D
        }
        self.job_details = {}                         # Store job-specific operation sequences
        self.workbench_queue = []                     # Queue for jobs waiting at the workbench

    def add_job(self, product_type):
        """Adds a new job to the conveyor if space is available."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Use a copy of the operations list so that each job’s operations are independent
        self.job_details[job_label] = self.product_operations[product_type][:]
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

        # If the entry is empty, pull a job from the buffer
        if self.buffer_jobs and self.conveyor[0] is None:
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates jobs based on a Poisson process."""
        num_new_jobs = np.random.poisson(self.arrival_rate)
        for _ in range(num_new_jobs):
            # Only add a job if the conveyor is not above its capacity
            if sum(x is not None for x in self.conveyor) < self.max_capacity * self.num_sections:
                product_type = random.choice(list(self.total_jobs.keys()))
                self.add_job(product_type)

    def transfer_to_workbench(self, index):
        """
        Transfers a job from a designated conveyor index to the workbench queue.
        (For example, the job in the last section is assumed to be ready for processing.)
        """
        if index < 0 or index >= self.num_sections:
            return
        if self.conveyor[index] is not None:
            job = self.conveyor[index]
            self.workbench_queue.append(job)
            print(f"Job {job} transferred from conveyor position {index} to the workbench queue.")
            self.conveyor[index] = None

    def display(self, agents):
        """Displays the current state of the conveyor, workbench queue, and agent workbenches."""
        conveyor_state = " <-> ".join([str(j) if j else "---" for j in self.conveyor])
        print(f"Conveyor: {conveyor_state}")
        jobs_on_conveyor = {job: self.job_details[job] for job in self.conveyor if job}
        print("Jobs on Conveyor:", jobs_on_conveyor)
        print("Buffer Jobs:", self.buffer_jobs)
        print("Workbench Queue:", self.workbench_queue)
        print("Agent Statuses:")
        for agent in agents:
            if agent.current_job:
                status = f"Busy with {agent.current_job} (remaining time: {agent.remaining_time})"
            else:
                status = "Idle"
            print(f"  Agent {agent.agent_id} on {agent.workbench}: {status}")

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.workbench = f"Workbench-{agent_id}"
        self.current_job = None
        self.remaining_time = 0

    def assign_job(self, job):
        """
        Assign a job to the agent. For simulation purposes, the processing
        time is randomly determined (e.g., between 1 and 3 time steps).
        """
        self.current_job = job
        self.remaining_time = random.randint(1, 3)
        print(f"Agent {self.agent_id} assigned job {job} on {self.workbench} for {self.remaining_time} time step(s).")

    def process(self):
        """
        Processes the current job for one time step.
        If processing is complete, returns the finished job; otherwise, returns None.
        """
        if self.current_job:
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                finished_job = self.current_job
                self.current_job = None
                print(f"Agent {self.agent_id} completed processing job {finished_job} on {self.workbench}.")
                return finished_job
        return None

# ----------------------- Simulation Setup -----------------------

# Create the conveyor (for example, with 10 sections)
conveyor = CircularConveyor(num_sections=10, max_capacity=0.8, arrival_rate=0.3)
# Create 3 agents (simulating 3 flexible workbenches)
agents = [Agent(i) for i in range(1, 4)]

# We designate one conveyor position as the "transfer point" for the workbench queue.
# Here we choose the last section (index = num_sections - 1).
workbench_index = conveyor.num_sections - 1

# ----------------------- Simulation Loop -----------------------

for timestep in range(10):
    print(f"\n--- Time Step {timestep + 1} ---")
    # 1. Generate new jobs on the conveyor.
    conveyor.generate_jobs()
    # 2. Move the conveyor.
    conveyor.move_conveyor()
    # 3. Transfer any job at the designated workbench position to the workbench queue.
    conveyor.transfer_to_workbench(workbench_index)
    
    # 4. Process jobs with agents.
    # First, let agents that are currently busy process their job.
    for agent in agents:
        if agent.current_job:
            finished_job = agent.process()
            if finished_job is not None:
                # When an agent finishes processing one operation:
                if finished_job in conveyor.job_details and conveyor.job_details[finished_job]:
                    # Remove the first (completed) operation.
                    completed_operation = conveyor.job_details[finished_job].pop(0)
                    print(f"Job {finished_job}: Operation {completed_operation} completed.")
                    # If the job still has pending operations, reinsert it into the conveyor.
                    if conveyor.job_details[finished_job]:
                        if conveyor.conveyor[0] is None:
                            conveyor.conveyor[0] = finished_job
                        else:
                            conveyor.buffer_jobs.append(finished_job)
                        print(f"Job {finished_job} reinserted into the conveyor for next operations: {conveyor.job_details[finished_job]}")
                    else:
                        print(f"Job {finished_job} is fully completed.")
    
    # Next, assign new jobs from the workbench queue to idle agents.
    for agent in agents:
        if agent.current_job is None and conveyor.workbench_queue:
            job_to_assign = conveyor.workbench_queue.pop(0)
            agent.assign_job(job_to_assign)
    
    # 5. Display the current state of the system.
    conveyor.display(agents)

# Final summary
print("\n--- Simulation Complete ---")
print("Total Jobs Generated:", conveyor.total_jobs)
print("Remaining Buffer:", conveyor.buffer_jobs)
print("Remaining Workbench Queue:", conveyor.workbench_queue)
