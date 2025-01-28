import gymnasium as gym
import pygame
import numpy as np

# Inisialisasi pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander Controller")

# Inisialisasi lingkungan
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=0, turbulence_power=0, render_mode='human')

# Variabel untuk throttle
main_throttle = 0.0
lateral_throttle = 0.0

def get_action(keys):
    """
    Mengonversi input keyboard menjadi aksi kontinu untuk LunarLander.
    W: +1 untuk throttle utama
    S: 0 untuk throttle utama
    A: +1 untuk throttle lateral (ke kanan)
    D: -1 untuk throttle lateral (ke kiri)
    Tidak ada tombol: semua throttle = 0
    """
    global main_throttle, lateral_throttle

    # Reset throttle
    main_throttle = 0.0
    lateral_throttle = 0.0

    # Periksa throttle utama
    if keys[pygame.K_w]:
        main_throttle = 1.0
    elif keys[pygame.K_s]:
        main_throttle = 0.0

    # Periksa throttle lateral
    if keys[pygame.K_a]:
        lateral_throttle = 1.0  # Ke kanan
    elif keys[pygame.K_d]:
        lateral_throttle = -1.0  # Ke kiri

    return np.array([main_throttle, lateral_throttle], dtype=np.float32)


# Loop utama
running = True
while running:
    state, info = env.reset()
    done = False
    total_reward = 0  # Untuk menyimpan total reward dalam satu episode
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                done = True

        if not running:
            break

        keys = pygame.key.get_pressed()
        action = get_action(keys)
        #print(f"Action: {action}")  # Menampilkan aksi throttle di setiap langkah
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward  # Tambahkan reward saat ini ke total reward
        # print(f"Step Reward: {reward}")  # Cetak reward setiap step

        env.render()

    print(f"Episode Total Reward: {total_reward}")  # Cetak total reward setiap episode

# Tutup pygame dan lingkungan
pygame.quit()
env.close()
