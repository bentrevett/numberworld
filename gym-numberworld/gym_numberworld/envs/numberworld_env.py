import gym
import numpy as np
import random
import itertools

COLORS = {'red': (0, 0),
          'green': (1, 1),
          'blue': (2, 2),
          'yellow': (0, 1),
          'purple': (0, 2),
          'cyan': (1, 2),
          'white': (0, 1, 2)}

SPRITES = {
            '@': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,1,1,1,1,0],
                           [0,1,1,1,1,1,0],
                           [0,1,1,1,1,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '0': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '1': np.array([[0,0,0,0,0,0,0],
                           [0,0,0,1,0,0,0],
                           [0,0,0,1,0,0,0],
                           [0,0,0,1,0,0,0],
                           [0,0,0,1,0,0,0],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '2': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '3': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '4': np.array([[0,0,0,0,0,0,0],
                           [0,1,0,0,0,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '5': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '6': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '7': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '8': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,0,0]
                          ]),
            '9': np.array([[0,0,0,0,0,0,0],
                           [0,1,1,1,1,1,0],
                           [0,1,0,0,0,1,0],
                           [0,1,1,1,1,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0]
                          ])}

class NumberWorldEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 grid_size = 10,
                 n_objects = 10,
                 time_limit = 250,
                 fog_size = None,
                 fog_type = None,
                 positive_reward = 1,
                 neutral_reward = 0,
                 negative_reward = -1,
                 ):

        assert grid_size >= 2, f'grid_size ({grid_size}) too small, must be at least 2'
        assert grid_size**2 >= (n_objects + 1), f'grid_size ({grid_size}) not big enough for desired n_objects ({n_objects})'
        assert n_objects >= 1, f'n_objects ({n_objects}) must be at least 1'
        assert time_limit >= 1, f'time_limit ({time_limit}) must be at least one'
        assert fog_size is None or (fog_size >= 1 and fog_type is not None), f'fog_size ({fog_size}) can only be set if fog_type ({fog_type}) is not None'
        assert fog_type in ['gray', 'noise', None], f'fog_type ({fog_type}) must be gray, noise or None'

        self.grid_size = grid_size
        self.n_objects = n_objects
        self.time_limit = time_limit
        self.fog_size = fog_size
        self.fog_type = fog_type
        self.positive_reward = positive_reward
        self.neutral_reward = neutral_reward
        self.negative_reward = negative_reward

        self.sprite_size = 7
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        numbers = [str(i) for i in range(10)]
        self.objects = list(itertools.product(colors, numbers)) 
        self.stoi = {x: i for i, x in enumerate(colors+numbers)}
        self.itos = {i: x for i, x in enumerate(colors+numbers)}

        assert n_objects <= len(self.objects), f'Maximum n_objects is {len(self.objects)}'

        grid_positions = [x for x in range(grid_size)]
        self.xy_positions = list(itertools.product(grid_positions, grid_positions))

        image_space = gym.spaces.Box(low=0, 
                                     high=1.0, 
                                     shape=(grid_size * 7, 
                                            grid_size * 7, 
                                            3), 
                                     dtype=np.float64)

        instruction_space = gym.spaces.MultiDiscrete([len(self.stoi), len(self.stoi)])

        self.observation_space = gym.spaces.Tuple((image_space, instruction_space))

        self.action_space = gym.spaces.Discrete(4)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        
        self.done = False
        self.t = 0

        agent_xy, *object_xys = random.sample(self.xy_positions, self.n_objects + 1)

        self.agent_x, self.agent_y = agent_xy

        self.object_xys = set(object_xys)

        random.shuffle(self.objects)
        env_objects = self.objects[:self.n_objects]

        self.env_objects = [(oc, on, ox, oy) for (oc, on), (ox, oy) in zip(env_objects, object_xys)]

        target_color, target_number, self.target_x, self.target_y = random.choice(self.env_objects)
        self.instruction = np.array([self.stoi[target_color], self.stoi[target_number]])

        if self.fog_type is not None:
            self.fog = self.make_fog()

        self.observation = self.draw_grid()

        return (self.observation, self.instruction)

    def draw_grid(self):

        grid = np.zeros((self.grid_size * self.sprite_size,
                         self.grid_size * self.sprite_size,
                         3))

        self.draw_sprite(grid, self.agent_x, self.agent_y, 'white', '@')

        for (object_color, object_number, object_x, object_y) in self.env_objects:
            grid = self.draw_sprite(grid, object_x, object_y, object_color, object_number)

        if self.fog_type is not None:
            grid = self.draw_fog(grid, self.fog, self.agent_x, self.agent_y)

        return grid

    def make_fog(self):

        if self.fog_type == 'noise':
            fog = np.random.normal(size=(self.grid_size * self.sprite_size,
                                         self.grid_size * self.sprite_size,
                                         3))
        elif self.fog_type == 'gray':
            fog = np.zeros(shape=(self.grid_size * self.sprite_size,
                                  self.grid_size * self.sprite_size,
                                  3))
            fog.fill(0.5)
        else:
            assert self.fog_type is None, f'Fog type must be noise, gray or None, got {self.fog_type}'

        return fog 

    def draw_fog(self, grid, fog, x, y):

        fog = np.copy(fog)

        fog_x_start = max(0, (x - self.fog_size) * self.sprite_size)
        fog_x_end = (x + self.fog_size + 1) * self.sprite_size
        fog_y_start = max(0, (y - self.fog_size) * self.sprite_size)
        fog_y_end = (y + self.fog_size + 1) * self.sprite_size

        fog[fog_x_start:fog_x_end, fog_y_start:fog_y_end, :] = grid[fog_x_start:fog_x_end:, fog_y_start:fog_y_end, :] 

        return fog

    def draw_sprite(self, grid, x, y, color, sprite):

        assert color in COLORS
        assert sprite in SPRITES

        x_start = x * self.sprite_size
        x_end = (x + 1) * self.sprite_size
        y_start = y * self.sprite_size
        y_end = (y + 1) * self.sprite_size

        grid[x_start:x_end,y_start:y_end,COLORS[color]] =  np.expand_dims(SPRITES[sprite], -1)

        return grid

    def step(self, action):
        
        assert not self.done, 'Tried to step when environment is done! Call env.reset()'
        assert action >= 0 and action <= 3, f'Action ({action}) should be between [0, 3]'

        if action == 0:
            #move up
            if (self.agent_x - 1) < 0:
                pass
            else:
                self.agent_x -= 1
            
        elif action == 1:
            #move right
            if (self.agent_y + 1) >= self.grid_size:
                pass
            else:
                self.agent_y += 1

        elif action == 2:
            #move down
            if (self.agent_x + 1) >= self.grid_size:
                pass
            else:
                self.agent_x += 1

        elif action == 3:
            #move left
            if (self.agent_y - 1) < 0:
                pass
            else:
                self.agent_y -= 1

        else:
            raise ValueError(f'Actions should be [0, 3], got {action}')

        #check if we've found an object

        if (self.agent_x, self.agent_y) in self.object_xys:
            self.done = True
            
            if (self.agent_x, self.agent_y) == (self.target_x, self.target_y):
                reward = self.positive_reward

            else:
                reward = self.negative_reward

        else:
            reward = self.neutral_reward

        self.observation = self.draw_grid()

        self.t += 1

        if self.t >= self.time_limit and not self.done:
            self.done = True
            reward = self.negative_reward

        return (self.observation, self.instruction), reward, self.done, None

    def render(self, mode='human', close= False):
        return (self.observation, self.instruction)