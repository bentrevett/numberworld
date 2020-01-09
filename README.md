# NumberWorld

*"To perform tasks specified by natural language instructions, autonomous  agents  need  to  extract  semantically  meaningful  representations  of  language  and  map  it  to  visual  elements and actions in the environment. This problem is called task-oriented  language  grounding"* - [Gated-Attention Architectures for Task-Oriented Language Grounding
](https://arxiv.org/abs/1706.07230)

--- 

This repository contains **NumberWorld**, a toy environment for task-oriented language grounding, aimed at being used with reinforcement learning. It was inspired by [this](https://arxiv.org/abs/1706.07230) paper, using a simple gridworld-like environment instead of the more computationally demanding VizDoom environment.

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/observation.png">
</p>

The agent is represented by a white square and all of the *objects* in the environments are represented by colored numbers. The objects can be one of 6 colors - red, blue, green, yellow, purple, cyan - and a number between 0 and 9.

The state is a (I, O) tuple, where I is the instruction and O is the observation. The instruction consists of two words, a color and a number, e.g. "blue 5". The observation is the raw pixel representation of the environment. The instruction has already been numericalized and the dictionary can be accessed via the environment's `itos` attribute. The state is 10x10 (by default) grid represented by a 70x70 pixel image (each cell is 7x7 pixels).

The goal of the agent is to reach the object denoted by the instruction, within the time-limit, whilst avoiding the incorrect objects. The agent's actions are moving a single cell north, south, east or west. The agent receives a positive reward for touching the correct object (+1 by default), a negative reward for touching an incorrect object or reaching the time-limit (-1 reward and 250 time-steps by default) and a neutral reward at all other times (0 by default). If the agent tries to go off the grid, it receives a neutral reward and stays where it is. The episode ends whenever an object is touched or the time-limit is reached.

In each episode, the agent's starting position and the objects (10 by default) within the environment are selected at random. Not all colors appear in each environment, e.g. one environment might have 5 red objects and 5 blue objects and another might have 2 red, 2 blue, 2 green, 2 yellow and 2 purple objects. The desired object from the instruction is always guaranteed to be within the environment.

The environment can also be made partially observable with the `fog_type` and `fog_size` parameters. With `fog_size` set to 2 the agent can only see the surrounding 2 cells in all directions, with the rest obscured by *fog*. A `fog_type` and `fog_size` of `None` (the default) means there is no fog and the environment is fully observable.

There are two types of fog: `gray` and `noise`. When the fog is made by noise, it is random each episode but will stay the same throughout the episode. Each are shown below with a `fog_size` of 2.

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/fog-gray.png">
</p>

<p align="center">
    <img src="https://github.com/bentrevett/rl-grounding/blob/master/fog-noise.png">
</p>

A successful agent must *recognize* objects in the raw pixel state, *explore* the environment when objects are obscured by fog, *ground* each concept in the instruction to actions and visual elements, and *navigate* to the correct object.

A random agent is provided, which reaches the correct object ~10% of the time with the default environment settings.

# Installation

```bash
cd gym-numberworld
pip install -e .
```