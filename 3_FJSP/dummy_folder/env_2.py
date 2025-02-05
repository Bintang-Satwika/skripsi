import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from circular_conveyor_1 import CircularConveyor

# ============================================================================
# Agent: Represents a processing unit (robot/workbench) with unique speed coefficients.
# ============================================================================
class Agent:
    def __init__(self, agent_id, speed_coefficients):
        self.agent_id = agent_id
        self.workbench = f"Workbench-{agent_id}"
        self.current_job = None
        self.remaining_time = 0
        # speed_coefficients is an array [v_op1, v_op2, v_op3] for this agent.
        self.speed_coefficients = speed_coefficients
    
    def process(self):
        """
        Processes the current job for one time step.
        If processing of the current operation is complete, returns the finished job.
        """
        if self.current_job:
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                finished_job = self.current_job
                self.current_job = None
                print(f"Agent {self.agent_id} finished job {finished_job}")
                return finished_job
        return None

# ============================================================================
# FJSPEnv: A Gymnasium environment integrating scheduling simulation, rule-based
# agent decisions, and observation/reward calculation.
# ============================================================================
class FJSPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, window_size=3, num_agents=3, max_steps=50):
        super(FJSPEnv, self).__init__()
        self.window_size = window_size        # Number of conveyor sections in agent's view (window)
        self.num_agents = num_agents          # Number of agents (robots)
        self.max_steps = max_steps            # Maximum simulation steps per episode
        
        # Conveyor parameters.
        self.num_conveyor_sections = 12
        self.max_conveyor_capacity = 0.75
        self.job_arrival_rate = 0.03
        
        # Create the circular conveyor.
        self.conveyor = CircularConveyor(num_sections=self.num_conveyor_sections,
                                         max_capacity=self.max_conveyor_capacity,
                                         arrival_rate=self.job_arrival_rate, num_agents=self.num_agents)
        # Define speed coefficients (v_r_l) for each agent per operation.
        # (Rows: agents; Columns: operations 1, 2, 3.)
        self.v_r_l = np.array([[1, 2, 3],
                                [1, 3, 5],
                                [2, 3, 4]])
        # Create agents with their respective speed coefficients.
        self.agents = [Agent(i+1, self.v_r_l[i]) for i in range(self.num_agents)]
        # Base processing times for operations.
        self.base_processing_times = {1: 3, 2: 2, 3: 4}
        
        # Gymnasium requires an action space; our actions are:
        # 0 = DECLINE, 1 = ACCEPT, 2 = WAIT, 3 = CONTINUE.
        self.action_space = spaces.Discrete(4)
        # Each agent’s observation vector has dimension (5 + window_size).
        obs_dim = 5 + self.window_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.num_agents, obs_dim),
                                            dtype=np.float32)
        
        self.step_count = 0
        # We use the last conveyor section as the designated transfer point.
        self.designated_transfer_index = self.num_conveyor_sections - 1

    def reset(self, seed=None, options=None):
        """Resets the environment and simulation components."""
        super().reset(seed=seed)
        self.step_count = 0
        self.conveyor = CircularConveyor(num_sections=self.num_conveyor_sections,
                                         max_capacity=self.max_conveyor_capacity,
                                         arrival_rate=self.job_arrival_rate, num_agents=self.num_agents)
        self.agents = [Agent(i+1, self.v_r_l[i]) for i in range(self.num_agents)]
        state = self._build_state()
        return state, {}

    def step(self, actions):
        """
        Expects a list of actions (one per agent) following the latest Gymnasium API.
        At every timestep each agent must select an action:
          - ACCEPT (1): If a job is present at the agent’s fixed position in the conveyor
            (or on its workbench) and the job's next operation is compatible with the agent,
            then the agent begins processing the job.
          - WAIT (2): Jika terdapat job di jendela (tetapi belum di posisi tetap),
            agen menunggu hingga job tersebut tiba di posisi tetap.
          - CONTINUE (3): Default, jika agen sedang memproses job atau tidak ada keputusan.
          - DECLINE (0): Jika job yang ada di posisi tetap tidak kompatibel, maka agen menolak job tersebut.
        """
        self.step_count += 1

        # (1) Generate new jobs.
        self.conveyor.generate_jobs()
        # (2) Move the conveyor.
        self.conveyor.move_conveyor()
        # (3) Transfer a job (if any) at the designated transfer point.
        self.conveyor.transfer_to_workbench(self.designated_transfer_index)
        
        # (4) Let agents process their current jobs.
        for agent in self.agents:
            finished_job = agent.process()
            if finished_job:
                # Upon finishing an operation, remove the first operation from the job.
                if finished_job in self.conveyor.job_details and self.conveyor.job_details[finished_job]:
                    completed_op = self.conveyor.job_details[finished_job].pop(0)
                    print(f"Job {finished_job}: Operation {completed_op} completed by Agent {agent.agent_id}")
                    # If the job has remaining operations, reinsert it into the conveyor.
                    if self.conveyor.job_details[finished_job]:
                        if self.conveyor.conveyor[0] is None:
                            self.conveyor.conveyor[0] = finished_job
                        else:
                            self.conveyor.buffer_jobs.append(finished_job)
                        print(f"Job {finished_job} reinserted with remaining ops {self.conveyor.job_details[finished_job]}")
                    else:
                        print(f"Job {finished_job} is fully completed.")
        
        # (5) Process actions for each agent.
        # Fixed positions for agents: 4, 8, and 12.
        fixed_positions = [4, 8, 12]
        for i, agent in enumerate(self.agents):
            # If the agent is busy, override action to CONTINUE.
            if agent.current_job is not None:
                continue
            action = actions[i]
            fixed_index = fixed_positions[i] - 1  # convert to 0-indexed
            if action == 1:  # ACCEPT
                if self.conveyor.conveyor[fixed_index] is not None:
                    job_to_assign = self.conveyor.conveyor[fixed_index]
                    # Remove the job from the conveyor.
                    self.conveyor.conveyor[fixed_index] = None
                    if self.conveyor.job_details[job_to_assign]:
                        current_op = self.conveyor.job_details[job_to_assign][0]
                        speed = agent.speed_coefficients[current_op - 1]
                        base_time = self.base_processing_times[current_op]
                        processing_time = max(1, int(np.round(base_time / speed)))
                    else:
                        processing_time = 0
                    agent.current_job = job_to_assign
                    agent.remaining_time = processing_time
                    print(f"Agent {agent.agent_id} ACCEPTS job {job_to_assign} (Op {current_op}) with processing time {processing_time}")
            elif action == 0:  # DECLINE
                if self.conveyor.conveyor[fixed_index] is not None:
                    declined_job = self.conveyor.conveyor[fixed_index]
                    self.conveyor.conveyor[fixed_index] = None
                    print(f"Agent {agent.agent_id} DECLINES job {declined_job} at fixed position.")
            elif action == 2:  # WAIT
                print(f"Agent {agent.agent_id} WAITS.")
            elif action == 3:  # CONTINUE
                print(f"Agent {agent.agent_id} CONTINUES (no decision required).")
        
        # (6) Build new observation state and compute reward.
        new_state = self._build_state()
        reward = self.reward_formula(new_state)
        done = (self.step_count >= self.max_steps)
        truncated = False
        info = {"actions": actions}
        return new_state, reward, done, truncated, info

    def _build_state(self):
        """
        Constructs an observation for each agent.
        Each agent’s state vector is defined as:
          [ y_r, O_r_a, O_r_b, O_r_t, Z_t, S_hat_y_r_t[0], ..., S_hat_y_r_t[window_size-1] ]
        where:
          - y_r: the fixed position (conveyor section) for the agent (4, 8, or 12)
          - O_r_a, O_r_b: example values for the first two operations the agent can perform
          - O_r_t: the next required operation of the job assigned (or 0 if idle)
          - Z_t: agent status (2 if working; 3 if idle but waiting jobs exist; 0 if idle with no job)
          - S_hat_y_r_t: the window of upcoming operations on the conveyor
        """
        state = np.zeros((self.num_agents, 5 + self.window_size), dtype=np.float32)
        # Fixed positions for agents.
        y_r = [4, 8, 12]
        O_r_a = [1, 2, 1]  # Example: Agent 1 can perform op1, Agent 2 op2, Agent 3 op1.
        O_r_b = [2, 3, 3]  # Example: Agent 1 can also perform op2, Agent 2 op3, Agent 3 op3.
        for i, agent in enumerate(self.agents):
            state[i, 0] = y_r[i]
            state[i, 1] = O_r_a[i] if i < len(O_r_a) else 0
            state[i, 2] = O_r_b[i] if i < len(O_r_b) else 0
            # O_r_t: current operation of the assigned job (if any)
            if agent.current_job and self.conveyor.job_details.get(agent.current_job, []):
                state[i, 3] = self.conveyor.job_details[agent.current_job][0]
            else:
                state[i, 3] = 0
            # Z_t: agent status (2 if working; 3 if idle and job available; 0 if idle and no job)
            if agent.current_job:
                state[i, 4] = 2
            else:
                # Jika ada job pada fixed position atau pada workbench, status adalah 3 (idle dengan job menunggu)
                fixed_index = (y_r[i] - 1) % self.num_conveyor_sections
                if self.conveyor.workbench_queue or (self.conveyor.conveyor[fixed_index] is not None):
                    state[i, 4] = 3
                else:
                    state[i, 4] = 0
            # S_hat_y_r_t: window of upcoming operations from the conveyor.
            start_index = (y_r[i] - 1) % self.num_conveyor_sections
            window_ops = []
            for j in range(self.window_size):
                idx = (start_index + j) % self.num_conveyor_sections
                job = self.conveyor.conveyor[idx]
                if job and self.conveyor.job_details.get(job, []):
                    window_ops.append(self.conveyor.job_details[job][0])
                else:
                    window_ops.append(0)
            state[i, 5:5+self.window_size] = window_ops
        return state

    def reward_formula(self, state, alpha=1.1, zeta=1, beta=1, gamma=0.5):
        """
        Computes a reward based on the agents' processing and waiting states.
        This reward uses the agents' speed coefficients (v_r_l) and state information.
        """
        R_step = 0
        R_process_all_agent = np.zeros(self.num_agents)
        R_wait_all = np.zeros(self.num_agents)
        
        sum_v_each_operation = np.sum(self.v_r_l, axis=0)
        mask_process = state[:, 4] == 2
        mask_process_index = np.where(mask_process)[0]
        a = np.where(state[mask_process_index, 3] == 1,
                     self.v_r_l[mask_process_index, 0] / sum_v_each_operation[0],
                     0)
        b = np.where(state[mask_process_index, 3] == 2,
                     self.v_r_l[mask_process_index, 1] / sum_v_each_operation[1],
                     0)
        c = np.where(state[mask_process_index, 3] == 3,
                     self.v_r_l[mask_process_index, 2] / sum_v_each_operation[2],
                     0)
        R_process_all_agent[mask_process_index] = a + b + c
        
        mask_wait = state[:, 4] == 3
        mask_wait_index = np.where(mask_wait)[0]
        x_distance = np.ones(len(mask_wait_index))
        a_wait = np.where(state[mask_wait_index, 3] == 1,
                          x_distance * self.v_r_l[mask_wait_index, 0] / sum_v_each_operation[0],
                          0)
        b_wait = np.where(state[mask_wait_index, 3] == 2,
                          x_distance * self.v_r_l[mask_wait_index, 1] / sum_v_each_operation[1],
                          0)
        c_wait = np.where(state[mask_wait_index, 3] == 3,
                          x_distance * self.v_r_l[mask_wait_index, 2] / sum_v_each_operation[2],
                          0)
        R_wait_all[mask_wait_index] = a_wait + b_wait + c_wait
        
        total_reward = -alpha + zeta * R_step + beta * R_process_all_agent + gamma * R_wait_all
        return float(np.sum(total_reward))
    
    def compute_rule_actions(self):
        """
        Computes a list of rule-based actions (one per agent) using First Come First Serve (FCFS).
        Untuk setiap agen:
          - Jika sedang sibuk, kembalikan CONTINUE (3).
          - Jika sedang idle, periksa *job* di posisi tetap (elemen pertama dari window).
              • Jika ada job di posisi tetap:
                  - Jika operasi job tersebut kompatibel dengan kapabilitas agen, kembalikan ACCEPT (1).
                  - Jika tidak kompatibel, kembalikan DECLINE (0).
              • Jika tidak ada job di posisi tetap tetapi ada job di sisa window, kembalikan WAIT (2)
                untuk menunggu job tersebut sampai tiba di posisi tetap.
              • Jika tidak ada job sama sekali, kembalikan CONTINUE (3).
        """
        state = self._build_state()
        actions = []
        # Hard-coded capabilities: Agent 1: {1,2}; Agent 2: {2,3}; Agent 3: {1,3}.
        capabilities = [{1, 2}, {2, 3}, {1, 3}]
        for i, agent in enumerate(self.agents):
            if agent.current_job is not None:
                actions.append(3)  # CONTINUE jika agen sedang sibuk.
            else:
                # Fixed position job adalah elemen pertama dari window.
                fixed_job_op = state[i, 5]
                # Sisa window (jika window_size > 1).
                rest_window = state[i, 6:5+self.window_size] if self.window_size > 1 else []
                if fixed_job_op != 0:
                    # Ada job di posisi tetap.
                    if fixed_job_op in capabilities[i]:
                        actions.append(1)  # ACCEPT
                    else:
                        actions.append(0)  # DECLINE
                elif np.any(rest_window != 0):
                    actions.append(2)  # WAIT
                else:
                    actions.append(3)  # CONTINUE (tidak ada job di view)
        return actions

    def render(self, mode="human"):
        """Renders the current simulation state to the console."""
        print(f"Step: {self.step_count}")
        self.conveyor.display()
        for agent in self.agents:
            status = (f"Working on {agent.current_job} with {agent.remaining_time} time left"
                      if agent.current_job else "Idle")
            print(f"Agent {agent.agent_id}: {status}")
        print("-" * 50)

# ============================================================================
# Main: Create and run the environment using Gymnasium format.
# ============================================================================
if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=20)
    obs, info = env.reset(seed=42)
    env.render()

    total_reward = 0
    done = False
    truncated = False
    while not done and not truncated:
        # Hitung aksi berbasis aturan First Come First Serve untuk semua agen.
        actions = env.compute_rule_actions()
        print("Actions chosen:", actions)
        obs, reward, done, truncated, info = env.step(actions)
        env.render()
        total_reward += reward
    print("Episode complete. Total Reward:", total_reward)
