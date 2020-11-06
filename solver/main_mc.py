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

parser = argparse.ArgumentParser()
parser.add_argument('--value_coeff', default=0.5, type=float)
parser.add_argument('--entropy_coeff', default=0.0, type=float)
parser.add_argument('--grad_clip', default=0.5, type=float)
parser.add_argument('--n_objects', default=1, type=int)
parser.add_argument('--grid_size', default=10, type=int)
parser.add_argument('--neutral_reward', default=0, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n_filters', default=64, type=int)
parser.add_argument('--emb_dim', default=64, type=int)
parser.add_argument('--hid_dim', default=256, type=int)
parser.add_argument('--lr', default=7e-4, type=float)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--n_episodes', default=100_000, type=int)
parser.add_argument('--norm_returns', default=0, type=int)
parser.add_argument('--norm_advantages', default=0, type=int)
args = parser.parse_args()

print(args)

wandb.init(project='numberworld', config=args, save_code=True)


def train(env, model, optimizer, device):
    model.train()
    log_prob_actions = []
    entropies = []
    value_preds = []
    rewards = []
    frames = []
    done = False
    episode_reward = 0
    (observation, instruction) = env.reset()
    while not done:
        frames.append(observation)
        observation = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0).permute(0, 3, 1, 2)
        instruction = torch.tensor(instruction, dtype=torch.long, device=device).unsqueeze(0)
        action_pred, value_pred = model(observation, instruction)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        entropy = dist.entropy()
        (observation, instruction), reward, done, _ = env.step(action.item())
        log_prob_actions.append(log_prob_action)
        entropies.append(entropy)
        value_preds.append(value_pred.squeeze(0))
        rewards.append(reward)
        episode_reward += reward
    frames.append(observation)
    log_prob_actions = torch.cat(log_prob_actions)
    entropies = torch.cat(entropies)
    value_preds = torch.cat(value_preds)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    returns, advantages = calculate_returns_advantages(rewards, value_preds, device)
    loss, policy_loss, value_loss, entropy = update_policy(model, advantages, log_prob_actions, returns,
                                                           value_preds, entropies, optimizer)
    rewards = rewards.mean().item()
    returns = returns.mean().item()
    advantages = advantages.mean().item()
    return loss, policy_loss, value_loss, entropy, episode_reward, rewards, returns, advantages, frames


def calculate_returns_advantages(rewards, values, device):
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in range(rewards.shape[0]-2, -1, -1):
        returns[t] = rewards[t] + returns[t+1] * args.discount_factor
    advantages = returns - values
    if args.norm_returns and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    if args.norm_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    return returns, advantages


def update_policy(model, advantages, log_prob_actions, returns, value_preds, entropies, optimizer):
    returns = returns.detach()
    policy_loss = -(advantages * log_prob_actions).mean()
    value_loss = F.mse_loss(returns, value_preds, reduction='mean')
    optimizer.zero_grad()
    loss = policy_loss + value_loss * args.value_coeff - entropies.mean() * args.entropy_coeff
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item(), entropies.mean().item()


env = gym.make('numberworld-v0',
               n_objects=args.n_objects,
               grid_size=args.grid_size,
               neutral_reward=args.neutral_reward)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

x = np.zeros(shape=env.observation_space[0].shape)
m = ImageProcessingModuleAlt(args.n_filters)
shape = m(torch.FloatTensor(x).unsqueeze(0).permute(0, -1, 1, 2))
print(shape.shape)
policy_input_dim = shape.numel()

vocab_size = len(env.itos)
n_actions = env.action_space.n

model = SolverAlt(args.n_filters, policy_input_dim, args.hid_dim, n_actions)

wandb.watch(model, log='all')

optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-5)

device = torch.device('cuda')
model = model.to(device)

episode_rewards = []

for episode in range(args.n_episodes):

    loss, policy_loss, value_loss, entropy, episode_reward, rewards, returns, advantages, frames = train(env, model, optimizer, device)

    frames = np.array(frames).transpose(0, -1, 1, 2)

    wandb.log({'loss': loss,
               'policy_loss': policy_loss,
               'value_loss': value_loss,
               'entropy': entropy,
               'episode_reward': episode_reward,
               'rewards': rewards,
               'returns': returns,
               'advantages': advantages})

    if episode % 100 == 0:
        wandb.log({f'episode_{episode}': wandb.Video(data_or_path=frames.astype(np.uint8))})

    episode_rewards.append(episode_reward)

    mean_episode_reward = np.mean(episode_rewards[-100:])

    print(f'episode: {episode:6}, episode_reward: {episode_reward:5.2f}, mean_episode_reward: {mean_episode_reward:5.2f}\r', end='')
