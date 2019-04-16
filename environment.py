import numpy as np
import random
import itertools
import collections
import matplotlib.pyplot as plt

str2array = {
            '@': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            ' ': [[0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0]
                 ],
            '0': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '1': [[0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0]
                 ],
            '2': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '3': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '4': [[0,0,0,0,0,0,0],
                  [0,1,0,0,0,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '5': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '6': [[0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0],
                  [0,1,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '7': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '8': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '9': [[0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,0],
                  [0,1,0,0,0,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0]
                 ],            
            }

colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
nums = [str(i) for i in range(10)]

str2idx = {x: i for i, x in enumerate(colors+nums)}
idx2str = {i: x for i, x in enumerate(colors+nums)}

class Environment:
    def __init__(self, 
                 grid_size = 10, 
                 n_objects = 10, 
                 pos_reward = 1, 
                 neg_reward = -1, 
                 neutral_reward = 0,
                 seed=None):

        self.grid_size = grid_size
        self.n_objects = n_objects
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self.neutral_reward = neutral_reward
        self.seed = seed

        self.done = False

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reset()

    def str_grid_to_np_grid(self):

        self.grid = np.zeros((self.grid_size*7, self.grid_size*7, 3))

        for i, row in enumerate(self.str_grid):

            for j, cell in enumerate(row):

                color, num = cell.split('-')

                obj = np.zeros((7,7,3))
            
                if color == 'red':
                    obj[:,:,0] = str2array[num]
                elif color == 'green':
                    obj[:,:,1] = str2array[num]
                elif color == 'blue':
                    obj[:,:,2] = str2array[num]
                elif color == 'white':
                    obj[:,:,0] = str2array[num]
                    obj[:,:,1] = str2array[num]
                    obj[:,:,2] = str2array[num]
                elif color == 'yellow':
                    obj[:,:,0] = str2array[num]
                    obj[:,:,1] = str2array[num]
                elif color == 'purple':
                    obj[:,:,0] = str2array[num]
                    obj[:,:,2] = str2array[num]
                elif color == 'cyan':
                    obj[:,:,1] = str2array[num]
                    obj[:,:,2] = str2array[num]
                else:
                    raise ValueError(f'{color} is not a valid color!')

                self.grid[i*7:(i+1)*7, j*7:(j+1)*7, :] = obj


    def reset(self):

        #generate all possible colour/object combinations
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        nums = [str(i) for i in range(10)]

        combinations = list(itertools.product(colors, nums))

        #want to have at least 2 of each color
        while True:

            #get N combinations at random
            random.shuffle(combinations)

            objects = combinations[:self.n_objects]

            counter = collections.Counter([c for c, n in objects])

            if all([x >= 2 for x in counter.values()]):
                break

        #create blank grid
        self.str_grid = [['white- ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        #pick starting position for agent
        self.agent_x, self.agent_y = np.random.randint(0, self.grid_size, 2)

        #place agent in starting position
        self.str_grid[self.agent_x][self.agent_y] = 'white-@'

        #keep track of all object positions
        self.object_positions = {(self.agent_x, self.agent_y)}

        #place all other objects
        for color, num in objects:
            object_x, object_y = self.agent_x, self.agent_y
            while (object_x, object_y) in self.object_positions:
                object_x, object_y = np.random.randint(0, self.grid_size, 2)
            self.object_positions.add((object_x, object_y))
            self.str_grid[object_x][object_y] = f'{color}-{num}'

        #remove agent position
        self.object_positions.remove((self.agent_x, self.agent_y))

        #convert to grid from string to np array
        self.str_grid_to_np_grid()
        
        #generate a go to <color> <num> question
        color, num = random.choice(objects)
        self.question = np.array([str2idx[color], str2idx[num]])
        
        return self.question, self.grid

    def step(self, action):

        assert not self.done, "Environment is done, you should reset it!"

        #actions: 0 = up, 1 = right, 2 = down, 3 = left

        prev_agent_x, prev_agent_y = self.agent_x, self.agent_y

        #move if possible
        if action == 0:
            if (self.agent_x - 1) < 0:
                pass
            else:
                self.agent_x -= 1

        elif action == 1:
            if (self.agent_y + 1) >= self.grid_size:
                pass
            else:
                self.agent_y += 1
        
        elif action == 2:
            if (self.agent_x + 1) >= self.grid_size:
                pass
            else:
                self.agent_x += 1

        elif action == 3:
            if (self.agent_y - 1) < 0:
                pass
            else:
                self.agent_y -= 1

        else:
            raise ValueError(f'Actions should be [0, 3], got: {action}')

        #check if we've found an object
        if (self.agent_x, self.agent_y) in self.object_positions:
            
            #if we have, set done to True
            self.done = True

            #what have we found?
            found_color, found_num = self.str_grid[self.agent_x][self.agent_y].split('-')

            #if we've found the correct object, give reward
            if str2idx[found_color] == self.question[0] and str2idx[found_num] == self.question[1]:
                reward = self.pos_reward
            else:
                #else give negative reward
                reward = self.neg_reward
        
        else:
            #if not found an object
            self.done = False
            reward = self.neutral_reward

        #update agent position in the grid
        self.str_grid[prev_agent_x][prev_agent_y] = 'white- '
        self.str_grid[self.agent_x][self.agent_y] = 'white-@'

        #redraw the grid
        self.str_grid_to_np_grid()

        return self.question, self.grid, reward, self.done
