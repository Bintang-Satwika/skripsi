import gymnasium as gym
from tqdm import tqdm
import pygame
import time
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
print(env.observation_space.shape, env.action_space.shape)

running = True
n_episodes = 2
action=[0.01,0.8] # [(roket mati <=0, roket naik>0), (-1 : belok kiri, 0 : lurus, 1 : belok kanan)]
iterasi=0
for episode in tqdm(range(n_episodes)):

    state, info = env.reset(seed=1)
    print(type(state))

    done = False
    print(episode)
    print(state)

    # play episode
    while not done:
        print("iterasi: ",iterasi)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                break
        if not running:
            break
        env.render()
        time.sleep(0.01)
        next_state, reward, terminated, truncated, info = env.step(action)
        # update if the environment is done and the current obs
        done = terminated or truncated
        state= next_state
        iterasi +=1

    print(f"Episode: {episode}, Total Reward:")

pygame.display.quit()
pygame.quit()
env.close()
