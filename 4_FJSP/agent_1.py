import numpy as np
import random
import math
# ============================================================================
# Agent: Merepresentasikan satu robot dengan workbench
# ============================================================================
class Agent:
    def __init__(self, agent_id: int, position: int, operation_capability : list, speed : float,  window_size: int, num_agent: int):
        
        self.window_size = window_size  # Ukuran window robot
        self.position = position  # Fixed position pada conveyor 
        self.operation_capability = operation_capability # Kemampuan operasi robot
        self.operation_now = 0   # Operasi yang sedang dikerjakan pada workbench
        self.status_all = [0]*num_agent  # Status robot (1: idle, 2: accepting job, 3: working, 4: completing job)
        self.workbench_remaining_operation = 0 # Sisa operasi yang harus dikerjakan pada workbench
        self.workbench_degree_of_completion=0    # Derajat penyelesaian operasi pada workbench
        self.remaining_operation=[0]*window_size # jumlah masing-masing operasi pada job dari  window robot terhadap conveyor
        self.processing_time_remaining = [0]*window_size # Waktu pemrosesan  yang tersisa pada window robot


        # Properti tetap
        self.many_operations= 2
        self.id = agent_id
        self.speed = speed
        #self.current_job = None
        self.processing_time_remaining = 0
        # self.pending_reinsertion = False

        # baru
        self.workbench = {}
        # self.window_product=[None]*window_size
        # self.buffer_job_to_workbench ={}
        # self.buffer_job_to_conveyor = {}

    def build_state(self):
        """
        Membangun representasi state sebagai numpy array.
        Vektor state: [position, operation_capability(2), operation_now, status(3), window(3)]
        Total dimensi = 1 + 2 + 1 + 3 + 3 = 10.
        """
        return np.hstack([
            np.array([self.position]),
            np.array(self.operation_capability),
            np.array([self.operation_now]),
            np.array(self.status_all),          # Perbaikan: tidak dibungkus list lagi
            np.array(self.workbench_remaining_operation),
            np.array(self.remaining_operation),
            np.array([self.processing_time_remaining])

        ])


    def processing_time(self, base_processing_time,speed):
        return math.ceil(base_processing_time / speed)


    def process(self, job_details):
       # print("job_details: ", job_details)
        #print("job_details[self.workbench]: ", job_details[self.workbench])
        # if self.workbench in job_details and len(job_details[self.workbench]) > 0:
        #         return job_details[self.workbench].pop(0)
        return 0
        
