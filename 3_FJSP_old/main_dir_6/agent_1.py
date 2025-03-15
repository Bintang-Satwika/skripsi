import numpy as np
import random
# ============================================================================
# Agent: Merepresentasikan robot dengan workbench
# ============================================================================
class Agent:
    def __init__(self, agent_id: int, position: int, operation_capability : list, speed : float,  window_size: int, num_agent: int):
        self.window_size = window_size  # Ukuran window
        '''
        Atribut state
        position (int), operation_capability (num_operation=2), operation_now (int), status_all (num_agents=3), 
        first_product_operation (window_size=3), second_product_operation (window_size=3), pick_product_window (int)
        '''

        self.position = position  # Fixed position pada conveyor (indeks 0-based)
        self.operation_capability = operation_capability
        self.operation_now = 0   # Operasi berikutnya (jika ada)
        self.status_all = [0]*num_agent  # Status robot (0: idle, 1: accepting job, 2: working, 3: completing job)
        self.first_product_operation=[0]*window_size
        self.second_product_operation=[0]*window_size
        self.pick_product_window = 0
        

        self.many_operations= 2

        # Properti tetap
        self.id = agent_id
        self.speed = speed
        #self.current_job = None
        self.processing_time_remaining = 0
        # self.pending_reinsertion = False

        # baru
        self.workbench = {}
        self.window_product=[None]*window_size
        self.buffer_job_to_workbench ={}
        self.buffer_job_to_conveyor = {}

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
            np.array(self.first_product_operation),
            np.array(self.second_product_operation),
            np.array([self.pick_product_window])

        ])


    def processing_time(self, base_processing_time):
        return int(np.ceil(base_processing_time / self.speed))


    def process(self, job_details):
       # print("job_details: ", job_details)
        #print("job_details[self.workbench]: ", job_details[self.workbench])
        if self.workbench in job_details and len(job_details[self.workbench]) > 0:
                return job_details[self.workbench].pop(0)
        
