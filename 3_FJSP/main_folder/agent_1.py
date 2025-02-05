import numpy as np
import random
import math
# ============================================================================
# Agent: Merepresentasikan robot dengan workbench
# ============================================================================
class Agent:
    def __init__(self, agent_id: int, position: int, operation_capability : list, speed : float, base_processing_time :float, window_size: int):
        # Atribut state
        self.position = position  # Fixed position pada conveyor (indeks 0-based)
        self.operation_capability = np.array(operation_capability)  # Kemampuan operasi (misal [1,2])
        self.operation_now = 0   # Operasi berikutnya (jika ada)
        self.status = 0          # 0 = idle, 1 = processing
        self.remaining_operation = np.zeros(window_size)  # Window observasi
        # Properti tetap
        self.id = agent_id
        self.speed = speed
        self.base_processing_time = base_processing_time
        self.current_job = None
        self.processing_time_remaining = 0
        self.pending_reinsertion = False

    def build_state(self):
        """
        Membangun representasi state sebagai numpy array.
        Vektor state: [position, operation_capability(2), operation_now, status, window(3)]
        Total dimensi = 1 + 2 + 1 + 1 + 3 = 8.
        """
        return np.hstack([
            np.array([self.position]),
            self.operation_capability,
            np.array([self.operation_now, self.status]),
            self.remaining_operation
        ])

    def processing_time(self):
        return np.array(np.ceil(self.base_processing_time / self.speed))

    def start_job(self):
        self.processing_time_remaining = math.ceil(self.base_processing_time / self.speed)
        self.status = 1  # Mulai memproses

    def process(self, job_details):
        """
        Mengurangi waktu pemrosesan setiap timestep.
        Jika waktu pemrosesan habis, operasi selesai.
        """
        if self.current_job is not None:
            if self.processing_time_remaining > 0:
                self.processing_time_remaining -= 1
                return None
            if self.processing_time_remaining == 0:
                # Selesaikan operasi: keluarkan operasi pertama dari job.
                if self.current_job in job_details and len(job_details[self.current_job]) > 0:
                    return job_details[self.current_job].pop(0)
        return None