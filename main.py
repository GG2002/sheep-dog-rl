from SheepDogEnv import SheepDogEnv
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal_init=False):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = 2*np.pi * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class BCNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(BCNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = np.pi/2*F.tanh(self.fc3(x))
        return x



def first_step_actor_init(param_path):
    first_step_actor = PolicyNetContinuous(
        state_dim, 128, action_dim).to(device)
    first_step_actor.load_state_dict(torch.load(param_path))
    first_step_actor.eval()
    return first_step_actor


def first_step_take_action(actor, state):
    state = torch.tensor(state, dtype=torch.float).to(device)
    mu, sigma = actor(state)
    action_dist = torch.distributions.Normal(mu, sigma)
    action = action_dist.sample()
    return [action.item()]


def BCNet_init(param_path):
    step_actor = BCNet(2, 128, action_dim).to(device)
    step_actor.load_state_dict(torch.load(param_path))
    step_actor.eval()
    return step_actor

def step_take_action(actor,state):
    state = torch.tensor(state[[0,3]], dtype=torch.float).to(device)

    # PPO
    # mu, sigma = step_actor(state)
    # print(mu,sigma)
    # action_dist = torch.distributions.Normal(mu, sigma)
    # action = action_dist.sample()
    # action = action.item()

    # BC
    action = actor(state).cpu().detach().numpy()[0]
    return [action]

if __name__ == "main":
    env = SheepDogEnv(circle_R=350, sheep_v=25, dog_v=80,
                    sec_split_n=10, store_mode=False, render_mode=True)
    state_dim = env._get_obs_array().shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    first_step_actor = first_step_actor_init(param_path="1step/params/arctor-1")
    # step_actor = BCNet_init(param_path="bc-2023-09-18-16-01-45")
    step_actor = BCNet_init(param_path="bc-2023-09-18-23-38-52")

    for i in range(2000):
        _st = env._get_obs_array()
        action = 0
        observation, reward, done, _, info = 0, 0, 0, 0, 0
        if(i == 0):
            _st[2] = (_st[2]-np.pi)/np.pi
            action = first_step_take_action(
                actor=first_step_actor, state=_st)[0] % (2*np.pi)
            observation, reward, done, _, info = env.step(
                action, first_step=True)  # 和环境交互
        else:
            _st[0]/=env.circle_R
            action = step_take_action(actor=step_actor,state=_st)[0]
            observation, reward, done, _, info = env.step(action)  # 和环境交互
        print(_st, action, reward)
        if done:
            # env.save()
            break
