import gymnasium as gym
import pygame
import numpy as np
import time  # Modul untuk melacak waktu

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
throttle_rate = 0.1  # Kecepatan perubahan throttle

# Timer untuk melacak waktu tombol terakhir ditekan
last_press_time = {
    "main": time.time(),  # Waktu terakhir throttle utama ditekan
    "lateral": time.time()  # Waktu terakhir throttle lateral ditekan
}
no_press_duration = 1  # Durasi (dalam detik) sebelum throttle tidak dieksekusi


def get_action(keys):
    """
    Mengonversi input keyboard menjadi aksi kontinu untuk LunarLander.
    Mengubah throttle secara bertahap seperti pedal gas mobil.
    """
    global main_throttle, lateral_throttle, throttle_rate, last_press_time

    current_time = time.time()

    # Periksa tombol throttle utama
    if keys[pygame.K_w]:
        main_throttle = min(main_throttle + throttle_rate, 1.0)  # Naik throttle utama
        last_press_time["main"] = current_time  # Perbarui waktu terakhir tombol ditekan
    elif keys[pygame.K_s]:
        main_throttle = max(main_throttle - throttle_rate, -1.0)  # Turun throttle utama
        last_press_time["main"] = current_time  # Perbarui waktu terakhir tombol ditekan
    else:
        if current_time - last_press_time["main"] < no_press_duration:
            main_throttle = max(0.0, main_throttle - throttle_rate * 1)  # Kembali ke 0 secara perlahan

    # Periksa tombol throttle lateral
    if keys[pygame.K_d]:
        lateral_throttle = max(lateral_throttle - throttle_rate, -1.0)  # Dorong ke kiri
        last_press_time["lateral"] = current_time  # Perbarui waktu terakhir tombol ditekan
    elif keys[pygame.K_a]:
        lateral_throttle = min(lateral_throttle + throttle_rate, 1.0)  # Dorong ke kanan
        last_press_time["lateral"] = current_time  # Perbarui waktu terakhir tombol ditekan
    else:
        if current_time - last_press_time["lateral"] < no_press_duration:
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

    step_counter = 0  # Menghitung langkah dalam 2 detik
    start_time = time.time()  # Waktu mulai 2 detik

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
        print("Action:", action)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward  # Tambahkan reward saat ini ke total reward
        step_counter += 1  # Hitung langkah

        # Periksa jika 2 detik telah berlalu
        if time.time() - start_time >= no_press_duration:
            print(f"Jumlah langkah dalam  detik: {step_counter}")
            start_time = time.time()  # Reset waktu mulai
            step_counter = 0  # Reset penghitung langkah

        env.render()

    print(f"Episode Total Reward: {total_reward}")  # Cetak total reward setiap episode

# Tutup pygame dan lingkungan
pygame.quit()
env.close()
