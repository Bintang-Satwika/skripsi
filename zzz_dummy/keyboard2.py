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

# Variabel untuk mengatur throttle
main_throttle = 0.0
lateral_throttle = 0.0
throttle_rate = 0.1 #Kecepatan perubahan throttle

# Timer untuk penghitungan lama tombol ditekan
pressed_keys = {
    pygame.K_w: False,
    pygame.K_s: False,
    pygame.K_a: False,
    pygame.K_d: False
}

def get_action(keys):
    """
    Mengonversi input keyboard menjadi aksi kontinu untuk LunarLander.
    Mengubah throttle secara bertahap seperti pedal gas mobil.
    """
    global main_throttle, lateral_throttle, throttle_rate

    # Periksa tombol throttle utama
    if keys[pygame.K_w]:
        main_throttle = min(main_throttle + throttle_rate, 1.0)  # Naik throttle utama
    elif keys[pygame.K_s]:
        main_throttle = max(main_throttle - throttle_rate, -1.0)  # Turun throttle utama
    else:
        main_throttle = max(0.0, main_throttle - throttle_rate * 1)  # Kembali ke 0 secara perlahan

    # Periksa tombol throttle lateral
    if keys[pygame.K_d]:
        lateral_throttle = max(lateral_throttle - throttle_rate, -1.0)  # Dorong ke kiri
    elif keys[pygame.K_a]:
        lateral_throttle = min(lateral_throttle + throttle_rate, 1.0)  # Dorong ke kanan
    else:  # Kembali ke 0 secara perlahan
        if lateral_throttle > 0:
            lateral_throttle = max(0.0, lateral_throttle - throttle_rate * 1)
        else:
            lateral_throttle = min(0.0, lateral_throttle + throttle_rate * 1)

    return np.array([main_throttle, lateral_throttle], dtype=np.float32)

# Loop utama
running = True
while running:
    state, info = env.reset()
    done = False
    total_reward = 0  # Untuk menyimpan total reward dalam satu episode
    main_throttle = 0.0
    lateral_throttle = 0.0
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
        done = terminated or truncated

        total_reward += reward  # Tambahkan reward saat ini ke total reward
       # print(f"Step Reward: {reward}")  # Cetak reward setiap step

        env.render()

    print(f"Episode Total Reward: {total_reward}")  # Cetak total reward setiap episode

# Tutup pygame dan lingkungan
pygame.quit()
env.close()