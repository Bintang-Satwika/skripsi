from env_5c_4_sekuensial import FJSPEnv
import numpy as np
import random
from tqdm import tqdm
from MASKING_ACTION_MODEL import masking_action

if __name__ == "__main__":
    env = FJSPEnv(window_size=3, num_agents=3, max_steps=600)
    for episode in tqdm(range(1, 1+ 10)):
        print("\nepisode:", episode)
        state, info = env.reset(seed=episode)
        reward_satu_episode = 0
        done = False
        truncated = False
        #print("\nEpisode:", episode)
        #print("Initial state:", state)
        while not done and not truncated:
            if len(env.conveyor.product_completed)>= env.n_jobs:
                print("All jobs are completed.")
                break

            mask_actions=masking_action(state, env)
            actions=[]

            for single, mask_action in zip(state, mask_actions):
                true_indices = np.where(mask_action)[0]
                random_actions = random.choice(true_indices)
                actions.append(random_actions)

            actions = np.array(actions)

            if None in actions:
                print("FAILED ACTION: ", actions)
                break

            next_state, reward, done, truncated, info = env.step(actions)
            reward_satu_episode += np.mean(reward)
            if env.FAILED_ACTION:
                print("episode:", episode)
                print("state:\n", state)
                print("actions:", actions)
                print("next_state:\n", next_state)
                #print(env.observation_all)
                #print("info:", info)
                print("FAILED ENV")
                break
            state = next_state
            #print("next_state:", next_state)

        print("Episode complete. Total Reward:", reward_satu_episode, "jumlah step:", env.step_count)
        order = {'A': 0, 'B': 1, 'C': 2}

        # Sorting by product type first, then by numeric value
        print("product completed: ",env.conveyor.product_completed)
        sorted_jobs = sorted(env.conveyor.product_completed, key=lambda x: (order[x[0]], int(x[2:])))

        #print("product sorted: ",sorted_jobs)
    print("selesai")