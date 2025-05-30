import numpy as np
import random

class CircularConveyor:
    def __init__(self, num_sections: int, max_capacity: float, arrival_rate: float, num_agents: int, n_jobs: int, current_episode_count: int):
        self.num_sections = num_sections            # Total number of sections
        self.max_capacity = max_capacity            # Maximum fill percentage (e.g., 0.75 means 75% fill)
        self.arrival_rate = arrival_rate            # Poisson arrival rate (jobs per time step)
        self.conveyor = [None] * num_sections       # Initialize empty conveyor
        self.buffer_jobs = []                       # Buffer for jobs when the entry is full
        self.total_jobs = {"A": 0, "B": 0, "C": 0}    # Job counters per product

        # Define the operation sequences for each product:
        #   - Product A: operations 1, 2
        #   - Product B: operations 2, 3
        #   - Product C: operations 1, 3
        self.product_operations = {
            "A": [1, 2],
            "B": [2, 3],
            "C": [1, 3],
        }
        self.job_index = 0 
        self.job_sequence = list(self.total_jobs.keys())
        self.job_details = {}  # For each job, stores its remaining operations.
        self.product_completed = []  # Buffer for finished products.
        self.n_jobs = n_jobs
        self.sum_n_jobs = 0
        self.num_agents = num_agents
        self.iteration = 0
        self.episode = current_episode_count

        self.episode_seed = 0

    def add_job(self, product_type: str):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        # According to the paper, if the entry (section 0) is available and capacity permits, add job immediately;
        # otherwise, add to the buffer.
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

        # If the first section is empty and the capacity conditions are met,
        # load a job from the buffer. (No extra random check is needed.)
        if (self.buffer_jobs and 
            sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None) > self.num_agents):
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates new jobs based on a Poisson process if below maximum total jobs."""
        if self.sum_n_jobs < self.n_jobs:
            self.iteration += 1
            # Sample the number of arrivals from a Poisson distribution
            num_arrivals = np.random.poisson(lam=self.arrival_rate)
            for _ in range(num_arrivals):
                # Check if we have reached the maximum number of jobs
                if self.sum_n_jobs >= self.n_jobs:
                    break

                # Select allowed product types (limit each product count to 7)
                allowed_types = [ptype for ptype in self.total_jobs if self.total_jobs[ptype] < 7]
                if not allowed_types:
                    break  # No product type is allowed, so stop generating jobs

                # Optionally, use a seed based on episode and iteration for reproducibility.
                random.seed(int(self.episode_seed + self.iteration))
                product_type = random.choice(allowed_types)
                self.sum_n_jobs += 1
                self.add_job(product_type)

    def display(self):
        """Displays the state of the conveyor, the buffer, and completed products."""
        conveyor_state = " <-> ".join([str(j) if j is not None else "---" for j in self.conveyor])
        print("Conveyor:", conveyor_state)
        print("Buffer:", self.buffer_jobs)
        print("Completed Products:", self.product_completed)

# # Example usage:
# if __name__ == "__main__":
#     # Create a CircularConveyor instance.
#     cc = CircularConveyor(num_sections=10, max_capacity=0.75, arrival_rate=0.03, num_agents=3, n_jobs=20, current_episode_count=1)
    
#     # Simulate a few time steps.
#     for _ in range(100):
#         cc.generate_jobs()    # Generate new jobs based on Poisson arrivals.
#         cc.move_conveyor()    # Move the conveyor.
#         cc.display()
#         print("-" * 40)
