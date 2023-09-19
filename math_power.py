from SheepDogEnv import SheepDogEnv
import torch
import numpy as np
import matplotlib.pyplot as plt

env = SheepDogEnv(circle_R=350, sheep_v=23, dog_v=80,
                  sec_split_n=5, store_mode=False, render_mode=True)
env.sheep_polar_coor=np.array([env.sheep_v,0.0])
env.dog_theta=np.array([0])%(np.pi*2)

for i in range(2000):
    _st = env._get_obs_array()
    L1 = _st[0]
    L2 = env.circle_R
    theta1 = _st[1]
    theta2 = _st[2]
    L3 = np.sqrt(L1**2 + L2**2 - 2*L1*L2*np.cos(theta2-theta1))
    theta3 = np.arcsin(L2/L3*np.sin(theta2-theta1))
    print(theta1,theta2,L3,theta3)
    if(np.abs(theta2-theta1)<np.arccos(L1/L2)):
        theta3=np.pi/2 if theta3>0 else -np.pi/2
    # action = env.action_space.sample()[0]
    action = theta3
    if(np.abs(_st[3]/env.dog_theta_v) > (env.circle_R-_st[0])/env.sheep_v + 1):
        action = 0
    observation, reward, done, _, info = env.step(action)  # 和环境交互
    print(_st,action, reward)
    if done:
        # env.save()
        break
