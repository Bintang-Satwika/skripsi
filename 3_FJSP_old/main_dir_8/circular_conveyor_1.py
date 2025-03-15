import numpy as np
import random
import math
# produk  tidak ada random poisson, namun product  A-B-C dirandom choice

class CircularConveyor:
    def __init__(self, num_sections: int, max_capacity: int, arrival_rate:float, num_agents: int, n_jobs: int, current_episode_count: int):
        self.num_sections = num_sections            # Total number of sections
        self.max_capacity = max_capacity            # Maximum fill percentage (e.g., 75%)
        self.arrival_rate = arrival_rate            # Poisson arrival rate for jobs
        self.conveyor = [None] * num_sections       # Initialize empty conveyor
        self.buffer_jobs = []                       # Buffer for jobs when the entry is full
        self.total_jobs = {"A": 0, "B": 0, "C": 0}    # Job counters per product
          # Job counters per product
        # Define the operation sequences for each product:
        #   - Product A: operations 1, 2, 3
        #   - Product B: operations 2, 3
        #   - Product C: operations 1, 2
        self.product_operations = {
            "A": [1,2],
            "B": [2,3],
            "C": [1,3],
        }
        dummy =[20,15,10]

        self.base_processing_times = {
            "A":[dummy[0], dummy[1],None],
            "B":[None,dummy[1]+5, dummy[2]+5],
            "C":[dummy[0]+10, None, dummy[2]+10],
        }
        self.job_index = 0 
        self.job_sequence = list(self.total_jobs.keys())
        self.job_details = {}  # For each job, stores its remaining operations.
        self.product_completed = []  # Buffer for finished products.
        self.n_jobs=n_jobs
        self.sum_n_jobs=0
        self.num_agents=num_agents
        self.iteration=0
        self.episode=current_episode_count

    def add_job(self, product_type: str):
        """Adds a new job (if capacity permits) with its product-specific operations."""
        self.total_jobs[product_type] += 1
        job_label = f"{product_type}-{self.total_jobs[product_type]}"
        # Copy the product's operation list so each job’s sequence is independent.
        self.job_details[job_label] = self.product_operations[product_type][:]
        self.buffer_jobs.append(job_label)

    def move_conveyor(self):
        """Moves jobs forward in a circular manner."""
        last_job = self.conveyor[-1]
        for i in range(self.num_sections - 1, 0, -1):
            self.conveyor[i] = self.conveyor[i - 1]
        self.conveyor[0] = last_job
        # If the first section is empty, load a job from the buffer.
        # np.random.seed(100+self.episode+self.iteration)
        # np.random.seed(2*self.episode+2*self.iteration)
        # print(" np.random.poisson:",  np.random.poisson(lam=0.8))
        #if   np.random.choice([True, False], p=[0.25, 0.75]) > 0:
       # if  np.random.poisson(lam=0.3)>0:
        if (self.buffer_jobs and 
            sum(1 for x in self.conveyor if x is not None) < self.max_capacity * self.num_sections and 
            self.conveyor[0] is None and 
            sum(1 for x in self.conveyor if x is None)> self.num_agents
            ) :
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """Generates new jobs based on a Poisson process (if below maximum capacity)."""
        if self.sum_n_jobs < self.n_jobs:
            self.iteration += 1
            random.seed(int(self.episode+2*self.iteration))
            if self.iteration % random.randint(10, 15)== 0:
                new_job = 1
            else:
                new_job = 0

            if new_job == 1:
                # Filter product types using the self.total_jobs counter (only choose if count is less than 7)
                allowed_types = [ptype for ptype in self.total_jobs if self.total_jobs[ptype] < 7]
                if not allowed_types:
                    # No product type is allowed (each has reached 7), so do nothing.
                    return
    
                random.seed(int(self.episode+2*self.iteration))
                product_type = random.choice(allowed_types)
                self.sum_n_jobs += 1
                self.add_job(product_type)


    def display(self):
        """Displays the state of the conveyor, the buffer, and completed products."""
        conveyor_state = " <-> ".join([str(j) if j is not None else "---" for j in self.conveyor])
        print("Conveyor:", conveyor_state)
        print("Buffer:", self.buffer_jobs)
        print("Completed Products:", self.product_completed)

