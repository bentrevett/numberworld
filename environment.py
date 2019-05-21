import numpy as np
import random
import itertools
import copy

str2array = {
            '@': [[0,0,0,0,0,0,0], #agent
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0]
                 ],
            '_': [[0,0,0,0,0,0,0], #blank space
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

sprite_size = 7
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
nums = [str(i) for i in range(10)]
object_combinations = list(itertools.product(colors, nums))

str2idx = {x: i for i, x in enumerate(colors+nums)}
idx2str = {i: x for i, x in enumerate(colors+nums)}

class Cell:
    """
    Represents the contents of a single cell
    """
    def __init__(self, color, object):
        
        assert color in colors + ['black', 'white']
        assert object in nums + ['@', '_']

        self.color = color
        self.object = object

class Environment:
    def __init__(self,
                 grid_size = 10,
                 observation_scale = 5,
                 n_objects = 10,
                 time_limit = 250,
                 fog_size = None,
                 fog_type = 'noise',
                 positive_reward = 1,
                 neutral_reward = 0,
                 negative_reward = -1,
                 seed = None,
                 ):

        assert grid_size >= 2 and (grid_size**2 + 1) >= n_objects and isinstance(grid_size, int)
        assert observation_scale >= 1 and isinstance(observation_scale, int)
        assert n_objects >= 1 and isinstance(n_objects, int)
        assert time_limit >= 1 and isinstance(time_limit, int)
        assert fog_size is None or (fog_size >= 1 and isinstance(fog_size, int))
        assert fog_type in ['gray', 'noise'] and isinstance(fog_type, str)
        assert isinstance(positive_reward, (int, float))
        assert isinstance(neutral_reward, (int, float))
        assert isinstance(negative_reward, (int, float))
        assert positive_reward > neutral_reward > negative_reward
        
        self.grid_size = grid_size
        self.observation_scale = observation_scale
        self.n_objects = n_objects
        self.time_limit = time_limit
        self.fog_size = fog_size
        self.fog_type = fog_type
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.neutral_reward = neutral_reward
        self.seed = seed
        
        self.done = True
        self.blank_grid = [[Cell('black', '_') for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        grid_positions = [x for x in range(self.grid_size)]
        self.xy_positions = list(itertools.product(grid_positions, grid_positions))

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self):

        self.generate_fog()

        self.done = False #tell when episode is over
        self.t = 0 #time-steps taken, used for time_limit

        #get random color-object combinations for this episode
        random.shuffle(object_combinations)
        objects = object_combinations[:self.n_objects]

        #make a blank grid
        self.grid = copy.deepcopy(self.blank_grid)

        #get agent starting position and object positions
        agent_xy, *object_xys = random.sample(self.xy_positions, self.n_objects+1)

        self.agent_x, self.agent_y = agent_xy

        #place agent in starting position
        self.grid[self.agent_x][self.agent_y] = Cell('white', '@')

        #keep track of all object positions for reward calculation
        self.object_positions = set()

        #place all objects, 
        for (obj_color, obj_num), (obj_x, obj_y) in zip(objects, object_xys):
            self.grid[obj_x][obj_y] = Cell(obj_color, obj_num)
            self.object_positions.add((obj_x, obj_y))

        #create observation from grid
        self.grid_to_observation()

        #generate a go to <color> <num> instruction
        color, num = random.choice(objects)
        self.instruction = np.array([str2idx[color], str2idx[num]])
        
        return self.instruction, self.observation

    def generate_fog(self):
        """
        Fog should not change during the episode
        This should be called by env.reset() and create the fog
        """

        if self.fog_size is not None:
            if self.fog_type == 'noise':
                self.fog = np.random.normal(size=(self.grid_size*sprite_size, self.grid_size*sprite_size, 3))
            elif self.fog_type == 'gray':
                self.fog = np.zeros((self.grid_size*sprite_size,self.grid_size*sprite_size,3))
                self.fog.fill(0.5)
            else:
                raise ValueError(f'Fog type {self.fog_type} not found')

    def grid_to_observation(self):

        #create observation array
        self.observation = np.zeros((self.grid_size*sprite_size, self.grid_size*sprite_size, 3))

        #place all objects
        for obj_x, obj_y in self.object_positions:
            
            #pixels for single cell
            cell_obs = np.zeros((sprite_size, sprite_size, 3))
            
            #get cell info
            cell = self.grid[obj_x][obj_y]

            #colour sprite
            if cell.color == 'red':
                cell_obs[:,:,0] = str2array[cell.object]
            elif cell.color == 'green':
                cell_obs[:,:,1] = str2array[cell.object]
            elif cell.color == 'blue':
                cell_obs[:,:,2] = str2array[cell.object]
            elif cell.color == 'yellow':
                cell_obs[:,:,0] = str2array[cell.object]
                cell_obs[:,:,1] = str2array[cell.object]
            elif cell.color == 'purple':
                cell_obs[:,:,0] = str2array[cell.object]
                cell_obs[:,:,2] = str2array[cell.object]
            elif cell.color == 'cyan':
                cell_obs[:,:,1] = str2array[cell.object]
                cell_obs[:,:,2] = str2array[cell.object]
            elif cell.color == 'white': #this should only be called on the end of an episode
                assert cell.object == '@'
                assert self.done == True
                cell_obs[:,:,0] = str2array[cell.object]
                cell_obs[:,:,1] = str2array[cell.object]
                cell_obs[:,:,2] = str2array[cell.object]
            else:
                raise ValueError

            #place sprite cell inside observation image
            self.observation[obj_x*sprite_size:(obj_x+1)*sprite_size, obj_y*sprite_size:(obj_y+1)*sprite_size, :] = cell_obs

        #draw and place agent
        cell_obs = np.zeros((sprite_size, sprite_size, 3))
        cell = self.grid[self.agent_x][self.agent_y]
        assert cell.color == 'white'
        assert cell.object == '@'
        cell_obs[:,:,0] = str2array[cell.object]
        cell_obs[:,:,1] = str2array[cell.object]
        cell_obs[:,:,2] = str2array[cell.object]
        self.observation[self.agent_x*sprite_size:(self.agent_x+1)*sprite_size, self.agent_y*sprite_size:(self.agent_y+1)*sprite_size, :] = cell_obs

        #draw fog
        if self.fog_size is not None:
            pos = list(range(self.grid_size))
            pos = list(itertools.product(pos, pos))
            xy_pos = [(x, y) for (x, y) in pos if (x > (self.agent_x + self.fog_size) or x < (self.agent_x - self.fog_size)) or (y > (self.agent_y + self.fog_size) or (y < self.agent_y - self.fog_size))]
            for x, y in xy_pos:
                self.observation[x*sprite_size:(x+1)*sprite_size,y*sprite_size:(y+1)*sprite_size] = self.fog[x*sprite_size:(x+1)*sprite_size,y*sprite_size:(y+1)*sprite_size]

        #scale up image
        scaled_observation = np.zeros((self.observation.shape[0] * self.observation_scale,
                                       self.observation.shape[1] * self.observation_scale,
                                       3))

        for j in range(self.observation.shape[0]):
            for k in range(self.observation.shape[1]):
                scaled_observation[j * self.observation_scale: (j+1) * self.observation_scale, 
                                   k * self.observation_scale: (k+1) * self.observation_scale, 
                                   :] = self.observation[j, k, :]

        self.observation = scaled_observation

    def step(self, action):

        assert not self.done, "Environment needs to be reset!"

        assert isinstance(action, int)

        prev_agent_x, prev_agent_y = self.agent_x, self.agent_y

        #actions: 0 = up, 1 = right, 2 = down, 3 = left

        #move if possible
        if action == 0: #up
            if (self.agent_x - 1) < 0:
                pass
            else:
                self.agent_x -= 1

        elif action == 1: #right
            if (self.agent_y + 1) >= self.grid_size:
                pass
            else:
                self.agent_y += 1
        
        elif action == 2: #down
            if (self.agent_x + 1) >= self.grid_size:
                pass
            else:
                self.agent_x += 1

        elif action == 3: #left
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
            found_cell = self.grid[self.agent_x][self.agent_y]

            found_object = found_cell.object
            found_color = found_cell.color

            #if we've found the correct object, give reward
            if str2idx[found_color] == self.instruction[0] and str2idx[found_object] == self.instruction[1]:
                reward = self.positive_reward
            else:
                #else give negative reward
                reward = self.negative_reward

        else:
            #if not found an object, give neutral reward
            reward = self.neutral_reward

        #update agent position in the grid
        self.grid[prev_agent_x][prev_agent_y] = Cell('black', '_')
        self.grid[self.agent_x][self.agent_y] = Cell('white', '@')

        #redraw the grid
        self.grid_to_observation()

        #update number of time-steps
        self.t += 1

        #if we have reached the maximum number of time-steps and still not found an object
        #then give negative reward
        if self.t >= self.time_limit and not self.done:
            self.done = True
            reward = self.negative_reward

        return self.instruction, self.observation, reward, self.done
