from SheepDogEnv import SheepDogEnv
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = np.pi / 2 * torch.tanh(self.fc_mu(x))
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


env = SheepDogEnv(circle_R=350, sheep_v=80, dog_v=80,
                  sec_split_n=5, store_mode=False, render_mode=True)
state_dim = env._get_obs_array().shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# PPO
# step_actor = PolicyNetContinuous(state_dim, 128, action_dim).to(device)
# step_actor.load_state_dict(torch.load("actor-4"))
# step_actor.eval()

# BC Learning
step_actor = BCNet(state_dim, 128, action_dim).to(device)
step_actor.load_state_dict(torch.load("bc-2023-09-18-16-01-45"))
step_actor.eval()


def step_take_action(state):
    state = torch.tensor(state, dtype=torch.float).to(device)

    # PPO
    # mu, sigma = step_actor(state)
    # print(mu,sigma)
    # action_dist = torch.distributions.Normal(mu, sigma)
    # action = action_dist.sample()
    # action = action.item()

    # BC
    action = step_actor(state).cpu().detach().numpy()[0]
    return [action]


# env.sheep_polar_coor=np.array([env.sheep_v,np.random.random()*np.pi*2])
# env.dog_theta=np.array([(env.sheep_polar_coor[1]-np.pi)%(np.pi*2)])
for i in range(2000):
    _st = env._get_obs_array()
    _st[0] /= env.circle_R
    # _st[1]=(_st[1]-np.pi)/np.pi
    # _st[2]=(_st[2]-np.pi)/np.pi
    action = step_take_action(_st)[0]
    observation, reward, done, _, info = env.step(action)  # 和环境交互
    print(_st, action, reward)
    if done:
        # env.save()
        break
