import random
from tqdm import tqdm

import environment

env = environment.Environment()

n_episodes = 1000
successful_episodes = 0

for episode in tqdm(range(n_episodes)):

    question, state = env.reset()
    done = False
    
    while not done:

        #pick random action
        action = random.randint(0, 3)

        #perform action
        instruction, state, reward, done = env.step(action)

        if done:
            if reward == env.pos_reward:
                successful_episodes += 1

print(f'Successfully found the correct object {successful_episodes/n_episodes*100}% of the time')
