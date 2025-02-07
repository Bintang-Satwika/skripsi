import numpy as np
import random
# ============================================================================
# Agent: Merepresentasikan robot dengan workbench
# ============================================================================
class Agent:
    def __init__(self, agent_id: int, position: int, operation_capability : list, speed : float,  window_size: int, num_agent: int):
        # Atribut state
        self.position = position  # Fixed position pada conveyor (indeks 0-based)
        self.operation_capability = operation_capability  # Kemampuan operasi (misal [1,2])
        self.operation_now = 0   # Operasi berikutnya (jika ada)
        self.status_all = [0]*num_agent  # Status robot (0: idle, 1: accepting job, 2: working, 3: completing job)

        self.remaining_operation = np.zeros(window_size)  # Window observasi
        # Properti tetap
        self.id = agent_id
        self.speed = speed
        #self.current_job = None
        self.processing_time_remaining = 0
        # self.pending_reinsertion = False

        # baru
        self.workbench = {}

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
            np.array(self.remaining_operation)
        ])


    def processing_time(self, base_processing_time):
        return int(np.ceil(base_processing_time / self.speed))

    # def start_job(self):
    #     self.processing_time_remaining = math.ceil(self.base_processing_time / self.speed)
        #self.status = 1  # Mulai memproses

    # def process(self, job_details):
    #     """
    #     Mengurangi waktu pemrosesan setiap timestep.
    #     Jika waktu pemrosesan habis, operasi selesai.
    #     """
    #     if self.current_job is not None:
    #         if self.processing_time_remaining > 0:
    #             self.processing_time_remaining -= 1
    #             return None
    #         if self.processing_time_remaining == 0:
    #             # Selesaikan operasi: keluarkan operasi pertama dari job.
    #             if self.current_job in job_details and len(job_details[self.current_job]) > 0:
    #                 return job_details[self.current_job].pop(0)
    #     return None

    def process(self, job_details):
       # print("job_details: ", job_details)
        #print("job_details[self.workbench]: ", job_details[self.workbench])
        if self.workbench in job_details and len(job_details[self.workbench]) > 0:
                return job_details[self.workbench].pop(0)
        
if __name__ == "__main__":
    agent =   Agent(
                agent_id=1,
                position=3,
                operation_capability=[1,2],
                speed=1,
                window_size=3,
                num_agent=3
            )
    print(agent.build_state())