import random
from tqdm import tqdm

import gym
import gym_numberworld

env = gym.make('numberworld-v0',
               grid_size = 10, # pass environment arguments to gym.make
               n_objects = 10,
               removed_objects = [('red', '3')]) # red 3 will not appear in environment

env.seed(1) #seed environment

n_episodes = 100_000
successful_episodes = 0

for episode in tqdm(range(n_episodes)):

    #environment must be reset at the start of every episode
    (observation, instruction) = env.reset()
    done = False

    while not done:

        #pick random action
        action = random.randint(0, 3)

        #perform action
        (observation, instruction) , reward, done, info = env.step(action)

        if done:
            if reward == env.positive_reward:
                successful_episodes += 1

success_percent = successful_episodes/n_episodes

print(f'Successfully found the correct object {success_percent*100:.3}% of the time')
