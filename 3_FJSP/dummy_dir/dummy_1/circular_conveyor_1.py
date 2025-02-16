import numpy as np
import random
class CircularConveyor:
    def __init__(self, num_sections, max_capacity, arrival_rate,  num_agents ):
        self.num_sections = num_sections            # Total number of sections
        self.max_capacity = max_capacity            # Maximum fill percentage (e.g., 75%)
        self.arrival_rate = arrival_rate            # Poisson arrival rate for jobs
        self.conveyor = [None] * num_sections       # Initialize empty conveyor
        self.buffer_jobs = []                       # Buffer for jobs when the entry is full
        self.total_jobs = {"A": 0, "B": 0, "C": 0}  # Job counters per product
        # Define the operation sequences for each product:
        #   - Product A: operations 1, 2, 3
        #   - Product B: operations 2, 3
        #   - Product C: operations 1, 2
        #   - Product D: operation 3
        self.product_operations = {
            "A": [1, 2, 3],
            "B": [2, 3],
            "C": [1, 2],
        }
        self.job_details = {}         # For each job, stores its remaining operations.
        self.workbench_queue = [False]* num_agents    # Workbench for each agents
    
    def add_job(self, product_type):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
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
        # If the first section is empty, pull a job from the buffer.
        if self.buffer_jobs and self.conveyor[0] is None:
            self.conveyor[0] = self.buffer_jobs.pop(0)
    
    def generate_jobs(self):
        """Generates new jobs based on a Poisson process (if below maximum capacity)."""
       # np.random.seed(10) DONT DELETE THIS LINE
        num_new_jobs = np.random.poisson(self.arrival_rate)
        for _ in range(num_new_jobs):
            # Only add a job if current fill is below max_capacity.
            if sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections:
                product_type = random.choice(list(self.total_jobs.keys()))
                self.add_job(product_type)
    
    def transfer_to_workbench(self, index):
        """
        Transfers a job from a designated conveyor index (e.g., the last section)
        to the workbench queue.
        """
        if index < 0 or index >= self.num_sections:
            return
        if self.conveyor[index] is not None:
            job = self.conveyor[index]
            self.workbench_queue.append(job)
            self.conveyor[index] = None

    def display(self):
        """Prints the state of the conveyor, its buffer, and the workbench queue."""
        conveyor_state = " <-> ".join([str(j) if j else "---" for j in self.conveyor])
        print("Conveyor:", conveyor_state)
        print("Buffer:", self.buffer_jobs)
        print("Workbench Queue:", self.workbench_queue)

# Example Simulation
conveyor = CircularConveyor(num_sections=12, max_capacity=0.75, arrival_rate=0.3, num_agents=3)

for timestep in range(10):  # Run simulation
    print(f"Time Step {timestep + 1}")
    conveyor.generate_jobs()
    conveyor.move_conveyor()
    conveyor.transfer_to_workbench(11)  
    conveyor.display()
    print()

print("Total Jobs Generated:", conveyor.total_jobs)
print("Buffer:", conveyor.buffer_jobs)
