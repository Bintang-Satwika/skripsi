import numpy as np
import random
import math


class CircularConveyor:
    def __init__(self, num_sections: int, max_capacity: int, arrival_rate:float, num_agents: int, n_jobs: int):
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
            "A": [1,2,3],
            "B": [2,3],
            "C": [1,3]
        }
        self.job_details = {}  # For each job, stores its remaining operations.
        self.product_completed = []  # Buffer for finished products.
        self.n_jobs=n_jobs
        self.sum_n_jobs=0
        self.num_agents=num_agents

    def add_job(self, product_type: str):
        """Adds a new job (if capacity permits) with its product-specific operations."""
       # print("   sum(1 for x in self.conveyor if x is None): ", sum(1 for x in self.conveyor if x is None))
        #print("   self.num_agents: ", self.num_agents)
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        #print("job_details:", self.job_details)
        # Insert the job into section 0 if available; otherwise, add to the buffer.
        if (sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None)> self.num_agents
            ) :
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
        if (self.buffer_jobs and 
            sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None)> self.num_agents
            ) :
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates new jobs based on a Poisson process (if below maximum capacity)."""
        # np.random.seed(10)  # DO NOT DELETE THIS LINE if reproducibility is required.
       # print("sum_n_jobs:", self.sum_n_jobs)
        #print("n_jobs:", self.n_jobs)
        if self.sum_n_jobs < self.n_jobs:
            new_job= min(1, np.random.poisson(self.arrival_rate, size=1)[0])
            #print("num_new_jobs:",   new_job)
            if new_job==1:
                    self.sum_n_jobs=self.sum_n_jobs+1
                    product_type = random.choice(list(self.total_jobs.keys()))
                    self.add_job(product_type)


    def display(self):
        """Displays the state of the conveyor, the buffer, and completed products."""
        conveyor_state = " <-> ".join([str(j) if j is not None else "---" for j in self.conveyor])
        print("Conveyor:", conveyor_state)
        print("Buffer:", self.buffer_jobs)
        print("Completed Products:", self.product_completed)

# # # Example Simulation
if __name__ == "__main__":
    conveyor = CircularConveyor(num_sections=12, max_capacity=0.75, arrival_rate=0.3, num_agents=3, n_jobs=20)

    for timestep in range(20):  # Run simulation
        print(f"Time Step {timestep + 1}")
        conveyor.move_conveyor()
        conveyor.generate_jobs()
        conveyor.display()
        print(conveyor.conveyor)
        print(conveyor.job_details.get("A-1"))
        print()

    print("Total Jobs Generated:", conveyor.total_jobs)
    print("Buffer:", conveyor.buffer_jobs)
