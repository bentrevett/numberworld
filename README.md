# Task-Oriented Language Grounding

*"To perform tasks specified by natural language instructions, autonomous  agents  need  to  extract  semantically  meaningful  representations  of  language  and  map  it  to  visual  elements and actions in the environment. This problem is called task-oriented  language  grounding"* - [Gated-Attention Architectures for Task-Oriented Language Grounding
](https://arxiv.org/abs/1706.07230)

--- 

This repository contains a toy environment for task-oriented language grounding, aimed at being used with reinforcement learning. It was inspired by [this](https://arxiv.org/abs/1706.07230) paper, using a simple gridworld-like environment instead of VizDoom.

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/state.png">
</p>

The agent is represented by the white square and all of the *objects* in the environments are represented by colored numbers. The objects can be one of 6 colors - red, blue, green, yellow, purple, cyan - and the numbers are 0 to 9.

The state is a (I, S) tuple, where I is the instruction and S is the state. The instruction consists of two words, a color and a number, e.g. "blue 5". The state is the raw pixel observation of the environment. The instruction has already been numericalized and the dictionary can be accessed via `environment.idx2str`. The state is 10x10 (by default) grid represented by a 70x70 pixel image (each cell is 7x7 pixels), which by default has been scaled up to 350x350 pixels.

The goal of the agent is to reach the object denoted by the instruction, within the time-limit, while avoiding the incorrect objects. The agent's actions are moving a single cell north, south, east or west. The agent receives a positive reward for touching the correct object (+1 by default), a negative reward for touching an incorrect object or reaching the time-limit (-1 reward and 250 time-steps by default) and a neutral reward at all other times (0 by default). If the agent tries to go off the grid, it receives a neutral reward and stays where it is. The episode ends whenever an object is touched or the time-limit is reached.

In each episode, the agent's starting position and the objects (10 by default) within the environment are selected at random. There are always at least two of each colored object within the environment, however not all colors appear in each environment, e.g. one environment might have 5 red objects and 5 blue objects and another might have 2 red, 2 blue, 2 green, 2 yellow and 2 purple objects. The desired object from the instruction is always within the environment.

The environment can also be made partially observable with the `fog_size` parameter. With `fog_size` set to 2 the agent can only see the surrounding 2 cells, with the rest obscured by *fog*. A `fog_size` of `None` (the default) means there is no fog and the environment is fully observable.

There are three types of fog: gray, noise-course and noise-fine. When the fog is made by noise, it is random each episode but will stay the same throughout the episode. Each are shown below.

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/noise-gray.png">
</p>

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/noise-course.png">
</p>

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/noise-fine.png">
</p>

A successful agent must *recognize* objects in the raw pixel state, *explore* the environment when objects are obscured by fog, *ground* each concept in the instruction to actions and visual elements, and *navigate* to the correct object.

A random agent is provided, which reaches the correct object ~10% of the time with the default settings.
