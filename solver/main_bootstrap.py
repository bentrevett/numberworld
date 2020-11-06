import argparse
import gym
import gym_numberworld  # NOQA
import numpy as np
import random
from solver import Solver, SolverAlt
from state_processing_module import ImageProcessingModule, ImageProcessingModuleAlt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import wandb


def discounted_cumsum(x, discount):
    y = torch.zeros_like(rewards)
    y[-1] = x[-1]
    for t in range(x.shape[0]-2, -1, -1):
        y[t] = x[t] + discount[t] * y[t+1]
    return y


def process_state(observation, instruction, device):
    observation = torch.FloatTensor(observation)
    instruction = torch.LongTensor(instruction)
    if len(observation.shape) < 4:
        observation = observation.unsqueeze(0)
    if len(instruction.shape) < 2:
        instruction = instruction.unsqueeze(0)
    observation = observation.permute(0, 3, 1, 2) / 255.0
    observation = observation.to(device)
    instruction = instruction.to(device)
    return observation, instruction


parser = argparse.ArgumentParser()
parser.add_argument('--value_coeff', default=0.5, type=float)
parser.add_argument('--entropy_coeff', default=0.01, type=float)
parser.add_argument('--grad_clip', default=1.0, type=float)
parser.add_argument('--n_objects', default=1, type=int)
parser.add_argument('--grid_size', default=10, type=int)
parser.add_argument('--neutral_reward', default=-0.01, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n_filters', default=64, type=int)
parser.add_argument('--emb_dim', default=64, type=int)
parser.add_argument('--hid_dim', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--n_steps', default=20, type=int)
parser.add_argument('--gae', default=0.95, type=float)
parser.add_argument('--norm_advantages', action='store_true')
parser.add_argument('--norm_returns', action='store_true')
parser.add_argument('--n_envs', default=2, type=int)
args = parser.parse_args()

wandb.init(project='numberworld', config=args, save_code=True)


def env_fn():
    env = gym.make('numberworld-v0',
                   n_objects=args.n_objects,
                   grid_size=args.grid_size,
                   neutral_reward=args.neutral_reward)
    return env


env = gym.vector.SyncVectorEnv([env_fn for _ in range(args.n_envs)])

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

x = np.zeros(shape=env.envs[0].observation_space[0].shape)
m = ImageProcessingModuleAlt(args.n_filters)
policy_input_dim = m(torch.FloatTensor(x).unsqueeze(0).permute(0, -1, 1, 2)).numel()

vocab_size = len(env.envs[0].itos)
n_actions = env.envs[0].action_space.n

#model = Solver(args.n_filters, vocab_size, args.emb_dim, policy_input_dim, args.hid_dim, n_actions)
model = SolverAlt(args.n_filters, policy_input_dim, args.hid_dim, n_actions)

wandb.watch(model, log='all')

optimizer = optim.Adam(model.parameters(), lr=args.lr)

device = torch.device('cuda')
model = model.to(device)

n_episodes = 0
episode_reward = 0
frames = []

(observation, instruction) = env.reset()
frames.append(observation)

while True:

    log_prob_actions = torch.zeros(args.n_steps, args.n_envs).to(device)
    values = torch.zeros(args.n_steps, args.n_envs).to(device)
    rewards = torch.zeros(args.n_steps, args.n_envs).to(device)
    dones = torch.zeros(args.n_steps, args.n_envs).to(device)
    entropies = torch.zeros(args.n_steps, args.n_envs).to(device)

    for step in range(args.n_steps):

        observation, instruction = process_state(observation, instruction, device)

        action_pred, value = model(observation, instruction)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        entropy = dist.entropy()

        (observation, instruction), reward, done, _ = env.step(action)
        frames.append(observation)
        episode_reward += reward

        n_episodes += np.sum(done)

        log_prob_actions[step] = log_prob_action
        values[step] = value.squeeze(-1)
        rewards[step] = torch.FloatTensor(reward)
        dones[step] = torch.FloatTensor(done)
        entropies[step] = entropy

    _observation, _instruction = process_state(observation, instruction, device)
    _, next_value = model(_observation, _instruction)
    next_value = next_value.squeeze(-1)
    rewards[-1] += (1 - dones[-1]) * args.discount_factor * next_value
    masked_discounts = args.discount_factor * (1 - dones)
    returns = discounted_cumsum(rewards, masked_discounts)

    if args.gae > 0:
        _values = torch.cat((values, next_value.unsqueeze(0)))
        deltas = rewards + masked_discounts * _values[1:] - _values[:-1]
        advantages = discounted_cumsum(deltas, args.gae * masked_discounts)
    else:
        advantages = returns - values

    if args.norm_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    if args.norm_returns:
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)

    policy_loss = -(advantages * log_prob_actions).mean()
    value_loss = F.smooth_l1_loss(returns, values)
    optimizer.zero_grad()
    loss = policy_loss + value_loss * args.value_coeff - entropies.mean() * args.entropy_coeff
    loss.backward()
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    wandb.log({'loss': loss.item(),
               'policy_loss': policy_loss.item(),
               'value_loss': value_loss.item(),
               'entropy': entropies.mean().item(),
               'n_episodes': n_episodes,
               'mean_rewards': rewards.mean().item()})
