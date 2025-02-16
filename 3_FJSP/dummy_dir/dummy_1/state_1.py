import numpy as np
# r adalah index dari agent
# i adalah index dari product
# j adalah index dari job
# k adalah index dari processing step
# l adalah index dari tipe operasi
# q adalah index dari conveyor section

# m adalah jumlah robot
# n adalah jumlah pekerjaan yang harus diselesaikan sebelum termination
# lambda adalah arrival rate dari job
# p adalah jumlah product
# o adalah jumlah unique operation types
# c adalah kapasitas conveyor belt section
# c_max adalah kapasitas conveyor belt dalam presentase


# t adalah index dari waktu
# y adalah index dari posisi agent
# v adalah kecepatan agent

c_conveyor_jumlah = 12 # jumlah conveyor belt
m_robot_jumlah = 3 # jumlah robot
n_job_jumlah = 20 # jumlah job
c_max = 0.75 # maksimal kapasitas conveyor dalam persentase


y_r=(5,5,5) # posisi dari agent (total ada 3 agent)
O_r=((1,2),(2,3),(1,3)) # seluruh operasi yang ada untuk setiap agent (total ada 3 agent)
O_r_t=(1,2,3) # operasi yang sedang dikerjakan oleh agent (total ada 3 agent)
Z_t=(0,1,2,3) # status agent (0 = idle, 1 = accepted, 2 = working,3 = done)

window_size=3
# sisa operasi yang harus dikerjakan pada waiting job di section q  dari conveyor belt oleh agent r pada waktu t
S_hat_y_r_t=(np.random.choice([1, 2], size=window_size),
             np.random.choice([2, 3], size=window_size),
             np.random.choice([1, 3], size=window_size)) 
v_r_l=(1,1,1) # kecepatan konstan agent r untuk operasi tipe Ol


state_all =[y_r, O_r, O_r_t, Z_t, S_hat_y_r_t]


# 1 = accepted, 2 = wait, 0 = decline, 3= continue
#  wait dari yr-1 sampai yr-window_size+1
action_all= [1] + [2] * (window_size - 1) + [0]+ [3]
print("A_all: ",action_all)

def reward_formula(alpha=1.1, zeta=1, beta=1, gamma=0.5):
    R_step = 0
    R_process = 0
    R_wait = 0

    if Z_1_t ==2:
        R_process = v_1_l/sum(v_r_l)
    
    if action_1==2:
        R_wait = x_section*v_1_l/sum(v_r_l)

    R_step= total_process_completed


    total_reward = -alpha + zeta*R_step + beta*R_process + gamma*R_wait
    return total_reward