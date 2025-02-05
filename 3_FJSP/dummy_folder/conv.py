import numpy as np
import random

class CircularConveyor:
    def __init__(self, num_sections=12, max_capacity=0.75, arrival_rate=0.03):
        self.num_sections = num_sections  # Number of conveyor sections
        self.max_capacity = max_capacity  # Max fill percentage
        self.arrival_rate = arrival_rate  # Poisson job arrival rate
        self.conveyor = [None] * num_sections  # Initialize empty conveyor
        self.buffer_jobs = []  # Buffer for excess jobs
        self.total_jobs = 0  # Total jobs generated

    def add_job(self, job_id):
        """ Adds a new job to the conveyor at the entry section if space is available """
        if self.conveyor[0] is None:  # job langsung masuk conveyor jika section pertama kosong
            self.conveyor[0] = f"Job-{job_id}"
        else:
            self.buffer_jobs.append(f"Job-{job_id}")  # job masuk ke buffer jika section pertama penuh

    def move_conveyor(self):
        """ Moves jobs forward in a circular manner """
        last_job = self.conveyor[-1]  # Save the last section job
        for i in range(self.num_sections - 1, 0, -1):  # Shift jobs forward
            self.conveyor[i] = self.conveyor[i - 1]
        self.conveyor[0] = last_job  # Move last job to the first section (circular movement)

        # jika conveyor section pertama kosong, job dipindahkan dari buffer ke conveyor
        if self.buffer_jobs and self.conveyor[0] is None:
            print("self.buffer: ",self.buffer_jobs)
            print("self.conveyor: ",self.conveyor)
            self.conveyor[0] = self.buffer_jobs.pop(0)

    def generate_jobs(self):
        """ Generates jobs based on a Poisson process """
        a_job = np.random.poisson(self.arrival_rate)  # Poisson-distributed arrivals
        for _ in range(a_job):
            print("sum:",sum(x is not None for x in self.conveyor))
            print("conve: ",[x is not None for x in self.conveyor])
            if sum(x is not None for x in self.conveyor) < self.max_capacity * self.num_sections:
                self.add_job(self.total_jobs+1)  # Nomor job dimulai dari 1
                self.total_jobs += 1

    def display(self):
        """ Print the conveyor state in a circular format """
        print(" <-> ".join([str(j) if j else "---" for j in self.conveyor]))

# Example Simulation
conveyor = CircularConveyor(num_sections=10, max_capacity=0.8, arrival_rate=0.3)

for timestep in range(200):  # Run simulation for 10 timesteps
    print(f"Time Step {timestep + 1}")
    conveyor.generate_jobs()
    conveyor.display()
    conveyor.move_conveyor()
    print()

print("self.total_jobs: ",conveyor.total_jobs)
print("self.buffer: ",conveyor.buffer_jobs)
