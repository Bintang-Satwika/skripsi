import gymnasium as gym
import pygame
import numpy as np

# Inisialisasi environment Gymnasium
env = gym.make("LunarLander-v3", render_mode="human")

# Inisialisasi Pygame untuk rendering teks
pygame.init()

# Konfigurasi font dan warna
font = pygame.font.SysFont("Arial", 24)
text_color = (255, 255, 255)  # Warna putih

# Reset environment
state, info = env.reset()

done = False
while not done:
    # Render environment
    env.render()

    # Event handler untuk keluar dari permainan
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    # Membuat teks
    text_surface = font.render("Hello, Gymnasium!", True, text_color)

    # Blit teks ke layar
    screen = pygame.display.get_surface()
    screen.blit(text_surface, (10, 10))  # Koordinat (x, y)

    # Update display
    pygame.display.flip()

    # Proses step random (hanya untuk demonstrasi)
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Tutup Pygame dan Gymnasium
pygame.quit()
env.close()
