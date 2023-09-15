from SheepDogEnv import SheepDogEnv
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, use_orthogonal_init=False):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_mu, gain=0.01)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # print(self.fc_mu(x))
        mu = 2*np.pi * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


env = SheepDogEnv(circle_R=350, sheep_v=80, dog_v=80,
                  sec_split_n=10, store_mode=False, render_mode=False)
state_dim = env._get_obs_array().shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

actor = PolicyNetContinuous(state_dim,128,action_dim).to(device)
actor.load_state_dict(torch.load("arctor-1"))
actor.eval()


def take_action(state):
    state = torch.tensor(state, dtype=torch.float).to(device)
    mu, sigma = actor(state)
    action_dist = torch.distributions.Normal(mu, sigma)
    action = action_dist.sample()
    return [action.item()]


env.reset()
for i in range(2000):
    _st = env._get_obs_array()
    _st[2] = (_st[2]-np.pi)/np.pi
    action = take_action(state=_st)[0]
    sheep_cur_theta = env.observation_space[1]
    if(sheep_cur_theta < np.pi/2 and action > sheep_cur_theta+np.pi*3/2):
        action -= 2*np.pi
    if(sheep_cur_theta > np.pi*3/2 and action < sheep_cur_theta-np.pi*3/2):
        action += 2*np.pi
    action = np.clip(
        action,
        sheep_cur_theta-np.pi/2,
        sheep_cur_theta+np.pi/2
    )
    # print(action)
    action %= (2*np.pi)
    observation, reward, done, _, info = env.step(action)  # 和环境交互
    if(i==0):
        print(reward)
    if done:
        env.save()
        break
