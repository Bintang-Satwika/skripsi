import numpy as np
import random

class CircularConveyor:
    def __init__(self, num_sections=12, max_capacity=0.75, arrival_rate=0.03):
        self.num_sections = num_sections  # Number of conveyor sections
        self.max_capacity = max_capacity  # Max fill percentage
        self.arrival_rate = arrival_rate  # Poisson job arrival rate
        self.conveyor = [None] * num_sections  # Initialize empty conveyor
        self.buffer_jobs = []  # Buffer for excess jobs
        self.total_jobs = {"A": 0, "B": 0, "C": 0, "D":0}  # Job counters for each product type
        self.product_operations = {
            "A": [1, 2, 3],  # Operations for Product A
            "B": [2, 3],     # Operations for Product B
            "C": [1, 2],      # Operations for Product C
            "D": [3]            # Operations for Product D

        }
        self.job_details = {}  # Store job-specific operation sequences

    def add_job(self, product_type):
        """ Adds a new job to the conveyor at the entry section if space is available """
        self.total_jobs[product_type] += 1  # Increase job counter for the selected product
        job_label = f"{product_type}-{self.total_jobs[product_type]}"  # Create unique job identifier
        self.job_details[job_label] = self.product_operations[product_type]  # Assign operations

        if self.conveyor[0] is None:  # Job enters conveyor if first section is empty
            self.conveyor[0] = job_label
        else:
            self.buffer_jobs.append(job_label)  # Job goes to buffer if first section is full

    def move_conveyor(self):
        """ Moves jobs forward in a circular manner """
        last_job = self.conveyor[-1]  # Save the last section job
        for i in range(self.num_sections - 1, 0, -1):  # Shift jobs forward
            self.conveyor[i] = self.conveyor[i - 1]
        self.conveyor[0] = last_job  # Move last job to the first section (circular movement)

        # Move a job from buffer to conveyor if the first section is empty
        if self.buffer_jobs and self.conveyor[0] is None:
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """ Generates jobs based on a Poisson process """
        a_job = np.random.poisson(self.arrival_rate)  # Poisson-distributed arrivals
        for _ in range(a_job):
            if sum(x is not None for x in self.conveyor) < self.max_capacity * self.num_sections:
                product_type = random.choice(list(self.total_jobs.keys()))  # Randomly select a product
                self.add_job(product_type)

    def display(self):
        """ Print the conveyor state in a circular format """
        conveyor_state = " <-> ".join([str(j) if j else "---" for j in self.conveyor])
        print(f"Conveyor: {conveyor_state}")
        print("Job Details:", {job: self.job_details[job] for job in self.conveyor if job})

# Example Simulation
conveyor = CircularConveyor(num_sections=10, max_capacity=0.8, arrival_rate=0.3)

for timestep in range(50):  # Run simulation
    print(f"Time Step {timestep + 1}")
    conveyor.generate_jobs()
    conveyor.display()
    conveyor.move_conveyor()
    print()

print("Total Jobs Generated:", conveyor.total_jobs)
print("Buffer:", conveyor.buffer_jobs)
