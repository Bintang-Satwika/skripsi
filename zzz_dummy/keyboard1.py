import gymnasium as gym
import pygame
import numpy as np

# Inisialisasi pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander Controller")

# Inisialisasi lingkungan
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')

def get_action(keys):
    """
    Mengonversi input keyboard menjadi aksi kontinu untuk LunarLander.
    """
    main_throttle = 0.0
    lateral_throttle = 0.0

    if keys[pygame.K_w]:
        main_throttle = 1.0  # Throttle utama penuh
    if keys[pygame.K_d]:
        lateral_throttle = -1.0  # Dorong ke kiri penuh
    if keys[pygame.K_a]:
        lateral_throttle = 1.0  # Dorong ke kanan penuh

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

        state, reward, terminated, truncated, info = env.step(action)
        print(state.shape)
        done = terminated or truncated

        total_reward += reward  # Tambahkan reward saat ini ke total reward
        #print(f"Step Reward: {reward}")  # Cetak reward setiap step

        env.render()

    print(f"Episode Total Reward: {total_reward}")  # Cetak total reward setiap episode

# Tutup pygame dan lingkungan
pygame.quit()
env.close()
