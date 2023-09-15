from SheepDogEnv import SheepDogEnv
from ppo import PPOContinuous
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

def test_agent(env,agent):
    test_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    for i in range(100):
        env.reset()
        _st = env._get_obs_array()
        _st[3:6]=0
        _st[2]=(_st[2]-np.pi)/np.pi
        act = agent.take_action(state=_st)[0]
        st, reward, done, _, _ = env.step(action=act)
        st[3:6]=0
        st[2]=(st[2]-np.pi)/np.pi
        test_dict["states"].append(_st)
        test_dict["actions"].append(act)
        test_dict["next_states"].append(st)
        test_dict["rewards"].append(reward)
        test_dict["dones"].append(False)
        # print(_st, act, reward, st)
    print("test:",np.mean(test_dict['rewards']))

env = SheepDogEnv(circle_R=350, sheep_v=70, dog_v=80,
                  sec_split_n=10, store_mode=True, render_mode=False)
torch.manual_seed(0)
state_dim = env._get_obs_array().shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.9
lmbda = 0.9
epochs = 1
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
agent = PPOContinuous(state_dim, hidden_dim, action_dim,
                      actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device,
                      use_orthogonal_init=True)
for ii in range(500):
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    for i in range(10):
        env.reset()
        _st = env._get_obs_array()
        _st[2]=(_st[2]-np.pi)/np.pi
        act = agent.take_action(state=_st)[0]
        st, reward, done, _, _ =  env.step(action=act)
        st[2]=(st[2]-np.pi)/np.pi
        transition_dict["states"].append(_st)
        transition_dict["actions"].append(act)
        transition_dict["next_states"].append(st)
        transition_dict["rewards"].append(reward)
        transition_dict["dones"].append(True)
    agent.update(transition_dict)
    if(ii%50==0):
        test_agent(env,agent)

torch.save(agent.actor.state_dict(),"params/arctor-1")
torch.save(agent.critic.state_dict(),"params/critic-1")