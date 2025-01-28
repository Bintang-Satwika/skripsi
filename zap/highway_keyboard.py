import gymnasium as gym
import highway_env
import numpy as np
from pynput import keyboard  # Library untuk mendeteksi input keyboard secara real-time
from highway_env.vehicle.kinematics import Vehicle

Vehicle.MIN_SPEED = 1  # Atur kecepatan minimum menjadi -20 m/s
Vehicle.MAX_SPEED = 20   # Atur kecepatan maksimum menjadi 50 m/s
# Konfigurasi environment
def create_env():
    config = {
        "manual_control": True,  # Aktifkan kontrol keyboard manual
        "action": {
            "type": "ContinuousAction",  # Mesin tetap menggunakan aksi kontinu
        },
        "lanes_count": 4,
        "vehicles_count": 10,
        "show_trajectories": True,
        "reward_speed_range": [20, 30],
        "collision_reward": -1,
    }
    return gym.make('highway-v0', render_mode='human', config=config)

# Buat environment
env = create_env()

# Variabel global untuk mencatat input keyboard
keyboard_input = None

# Fungsi untuk menangani penekanan tombol
def on_press(key):
    global keyboard_input
    try:
        if key == keyboard.Key.up:
            keyboard_input = [1.0, 0.0]  # Maksimal percepatan, tanpa kemudi
        elif key == keyboard.Key.down:
            keyboard_input = [-1.0, 0.0]  # Rem penuh, tanpa kemudi
        elif key == keyboard.Key.left:
            keyboard_input = [0.0, -1.0]  # Kemudi penuh kiri
        elif key == keyboard.Key.right:
            keyboard_input = [0.0, 1.0]  # Kemudi penuh kanan
    except AttributeError:
        pass

# Fungsi untuk menangani pelepasan tombol
def on_release(key):
    global keyboard_input
    # Reset aksi ke None saat tombol dilepaskan
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        keyboard_input = None

# Listener untuk keyboard
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Fungsi untuk aksi mesin (kontinu)
def get_machine_action():
    return [0.5, 0.0]  # Contoh: percepatan sedang, tanpa kemudi

# Loop utama
try:
    obs, info = env.reset()
    done, truncated = False, False

    while not (done or truncated):
        # Gunakan input keyboard jika ada
        action = keyboard_input

        if action is None:
            # Jika tidak ada input keyboard, gunakan aksi mesin
            action = get_machine_action()
            print("Kontrol: Mesin")
        else:
            print("Kontrol: Keyboard")

        # Langkah environment
        obs, reward, done, truncated, info = env.step(action)

        # Render environment
        env.render()

        # Informasi tambahan (opsional)
        print("Reward:", reward)
        print("Done:", done)
        print("Truncated:", truncated)

except KeyboardInterrupt:
    print("Program dihentikan.")
finally:
    env.close()
    listener.stop()
