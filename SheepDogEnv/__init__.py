from typing import Any, SupportsFloat
import matplotlib.pyplot as plt
import gymnasium as gym
import pygame
from pygame.locals import *
from pygame import gfxdraw
import numpy as np
import time
import os
from datetime import datetime


class SheepDogEnv(gym.Env):
    def __init__(self, circle_R, sheep_v, dog_v, sec_split_n, store_mode=True, render_mode=False) -> None:
        super().__init__()
        self.circle_R = circle_R
        # 采样间隔
        self.dt = 1/sec_split_n
        # 羊的速度
        self.sheep_v = sheep_v * self.dt
        # 犬的角速度
        self.dog_theta_v = dog_v / self.circle_R * self.dt
        # 羊的极坐标：[r,theta]
        self.sheep_polar_coor = [0,0]
        # self.sheep_polar_coor = [self.sheep_v, np.random.random()*2*np.pi]
        # 犬的极坐标角度列表，暂时训练一只犬
        self.dog_theta = np.random.uniform(0, 2*np.pi, 1)

        # 动作空间与状态空间
        self.action_space = gym.spaces.Box(
            low=-np.pi/2, high=np.pi/2, dtype=np.float32
        )
        self.observation_space = self._get_obs_array()

        # 存储相关参数
        self.store_mode = store_mode
        if(self.store_mode):
            # [(observation, action, reward, _observation)]
            self.store_data = []

        # 渲染相关参数
        self.render_mode = render_mode
        if(self.render_mode):
            self.render_diplay_size = (800, 800)
            self.original_x = int(self.render_diplay_size[0]/2)
            self.original_y = int(self.render_diplay_size[1]/2)
            self.img_size = (30, 30)

            pygame.init()
            self.screen = pygame.display.set_mode(self.render_diplay_size)
            pygame.display.set_caption('古老的羊-犬博弈')
            self.sheep_img = pygame.transform.scale(pygame.image.load(
                "./assets/sheep.jpg").convert(), self.img_size)
            self.dog_img = pygame.transform.scale(pygame.image.load(
                "./assets/dog.jpg").convert(), self.img_size)
            self.render()

    def _transform_polar_to_rendering_xy(self, r, theta):
        return self.original_x + r*np.cos(theta), self.original_y - r*np.sin(theta)

    def _get_obs(self):
        sheep_dog_between_theta=self.dog_theta-self.sheep_polar_coor[1]
        sheep_dog_between_theta[sheep_dog_between_theta>np.pi]-=2*np.pi
        sheep_dog_between_theta[sheep_dog_between_theta<-np.pi]+=2*np.pi
        return {
            "sheep_polar_coor_r": self.sheep_polar_coor[0],
            "sheep_polar_coor_theta": self.sheep_polar_coor[1],
            "dog_theta": self.dog_theta[0],
            "sheep_dog_between_theta":sheep_dog_between_theta[0],
        }

    def _get_obs_array(self):
        sheep_dog_between_theta=self.dog_theta-self.sheep_polar_coor[1]
        sheep_dog_between_theta[sheep_dog_between_theta>np.pi]-=2*np.pi
        sheep_dog_between_theta[sheep_dog_between_theta<-np.pi]+=2*np.pi
        return np.array([
            self.sheep_polar_coor[0],
            self.sheep_polar_coor[1],
            self.dog_theta[0],
            sheep_dog_between_theta[0],
        ])

    def _get_info(self):
        distance = np.sqrt(self.circle_R**2+self.sheep_polar_coor[0]**2-2*self.circle_R *
                           self.sheep_polar_coor[0]*np.cos(self.dog_theta-self.sheep_polar_coor[1]))
        return{
            "distance": distance
        }

    def _get_reward(self, _info, info, _ob, ob):
        s_d_between_theta = np.abs(_ob[1]-_ob[2])
        if s_d_between_theta > np.pi:
            s_d_between_theta = np.pi*2-s_d_between_theta
        s_d_between = s_d_between_theta/np.pi
        return (
            # 与上一步相比，羊与圆圈的距离减少了多少
            s_d_between * \
            (((self.circle_R - _ob[0]) - (self.circle_R - ob[0])))
            # 与上一步相比，羊与犬的距离增大或减少了多少
            + (1-s_d_between) * \
            (np.max(info["distance"])-np.max(_info["distance"]))
        )

    def _done(self):
        distance = self._get_info()["distance"]
        if(np.sum(distance[distance < self.dog_theta_v*self.circle_R]) > 1):
            return True, True
        if(self.sheep_polar_coor[0] > self.circle_R):
            return True, False
        return False, False

    def step(self, action: Any, first_step=False) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        _info = self._get_info()
        _observation_space = self._get_obs_array()
        # 先更新羊，再更新狗，最后判断狗是否能抓到羊
        sheep_next_x = 0
        sheep_next_y = 0
        if first_step:
            sheep_next_x = self.sheep_polar_coor[0]*np.cos(
                self.sheep_polar_coor[1])+self.sheep_v*np.cos(action)
            sheep_next_y = self.sheep_polar_coor[0]*np.sin(
                self.sheep_polar_coor[1])+self.sheep_v*np.sin(action)
        else:
            sheep_ds_theta = (self.sheep_polar_coor[1]-action) % (np.pi*2)
            sheep_next_x = self.sheep_polar_coor[0]*np.cos(
                self.sheep_polar_coor[1])+self.sheep_v*np.cos(sheep_ds_theta)
            sheep_next_y = self.sheep_polar_coor[0]*np.sin(
                self.sheep_polar_coor[1])+self.sheep_v*np.sin(sheep_ds_theta)
        self.sheep_polar_coor[0] = np.sqrt(sheep_next_x**2+sheep_next_y**2)
        if sheep_next_x > 0:
            self.sheep_polar_coor[1] = np.arctan(
                sheep_next_y/sheep_next_x) % (2*np.pi)
        else:
            self.sheep_polar_coor[1] = np.arctan(
                sheep_next_y/sheep_next_x)+np.pi
        # print("action:{:.2f},\tsheep(x,y):({:.2f},{:.2f}),\tsheep(r,theta):({:.2f},{:.2f}),\tsheep_ds_theta:{:.2f}".format(
        #       action, sheep_next_x, sheep_next_y, self.sheep_polar_coor[0], self.sheep_polar_coor[1], action))

        def get_dog_action(x):
            flag = 1
            ttmp = x-self.sheep_polar_coor[1]
            if(ttmp > 0):
                flag *= -1
            if(np.abs(ttmp) > np.pi):
                flag *= -1
            if(abs(ttmp) < self.dog_theta_v):
                return -ttmp
            return flag*self.dog_theta_v
        dog_ds = np.array(list(map(get_dog_action, self.dog_theta)))
        self.dog_theta = (self.dog_theta + dog_ds) % (2*np.pi)
        # print("dog_ds:({:.2f}),\tdog_theta:{:.2f}".format(
        #     dog_ds[0], self.dog_theta[0]))

        if(self.render_mode):
            time.sleep(self.dt)
            self.render()

        self.observation_space = self._get_obs_array()
        done, catched = (False, False) if (
            self.circle_R-self.sheep_polar_coor[0] > 10) else self._done()
        info = self._get_info()
        reward = self._get_reward(
            _info, info, _observation_space, self.observation_space)
        if(done):
            # print("Sheep is catched:", captched)
            if(catched):
                reward -= 1000
            else:
                reward += 500

        if(self.store_mode):
            self.store_data.append(
                (_observation_space, action, reward, self.observation_space)
            )
        return self.observation_space, reward, done, False, info

    def render(self):
        self.screen.fill((255, 255, 255))
        gfxdraw.aacircle(self.screen, self.original_x,
                         self.original_y, self.circle_R, (0, 0, 255))

        sheep_x, sheep_y = self._transform_polar_to_rendering_xy(
            self.sheep_polar_coor[0], self.sheep_polar_coor[1]
        )
        self.screen.blit(self.sheep_img,
                         (
                             sheep_x - self.img_size[0]/2,
                             sheep_y - self.img_size[1]/2
                         )
                         )
        pygame.draw.circle(surface=self.screen, center=(sheep_x, sheep_y),
                           radius=self.img_size[0]/6, color=(0, 152, 255))

        for d_theta in self.dog_theta:
            d_x, d_y = self._transform_polar_to_rendering_xy(
                self.circle_R, d_theta
            )
            self.screen.blit(self.dog_img,
                             (
                                 d_x - self.img_size[0]/2,
                                 d_y - self.img_size[1]/2
                             )
                             )
            pygame.draw.circle(surface=self.screen, center=(d_x, d_y),
                               radius=self.img_size[0]/6, color=(0, 113, 182))

        pygame.display.update()

        return

    def reset(self):

        if(self.store_mode):
            self.store_data = []
        # 羊的极坐标：[r,theta]
        self.sheep_polar_coor = [0, 0]
        # self.sheep_polar_coor = [self.sheep_v, np.random.random()*2*np.pi]
        # 犬的极坐标角度列表，暂时训练一只犬
        self.dog_theta = np.random.uniform(0, 2*np.pi, 1)

        return super().reset()

    def save(self, save_path=None):
        if(self.store_mode != True):
            print("store_mode is False! No store_data to save!")
            return
        if not (os.path.exists("./observation_store")):
            os.makedirs("./observation_store")
        if not (os.path.exists("./1step_store")):
            os.makedirs("./1step_store")
        fs = open(save_path, "a+")if save_path != None else os.open(
            "./observation_store/"+str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), os.O_APPEND)
        for data in self.store_data:
            print(str(data))
            fs.write(str(data))
            fs.write("\n")
        fs.close()


# env = SheepDogEnv(circle_R=350, sheep_v=70, dog_v=80,
#                   sec_split_n=10, store_mode=False, render_mode=False)
# # while True:
# action_list = []
# action1_list = []
# action2_list = []
# theta_list = []
# for i in range(2000):
#     action = env.action_space.sample()[0]
#     action_list.append(action)
#     print(action)
#     sheep_cur_theta = env.observation_space["sheep_polar_coor_theta"]
#     theta_list.append(sheep_cur_theta)
#     if(sheep_cur_theta < np.pi/2 and action > sheep_cur_theta+np.pi*3/2):
#         action -= 2*np.pi
#     if(sheep_cur_theta > np.pi*3/2 and action < sheep_cur_theta-np.pi*3/2):
#         action += 2*np.pi
#     if(i < 1):
#         action = 0.4*np.pi
#     if(i >= 1):
#         action = np.clip(
#             action,
#             sheep_cur_theta-np.pi/2,
#             sheep_cur_theta+np.pi/2
#         )
#     print(action)
#     action1_list.append(action)
#     action %= (2*np.pi)
#     action2_list.append(action)
#     print(action)
#     observation, reward, done, _, info = env.step(action)  # 和环境交互
#     if done:
#         env.save()
#         break

# # plt.hist(action_list, bins=20)
# # plt.hist(action1_list, bins=20)
# # plt.hist(action2_list, bins=20)
# plt.plot(action_list)
# plt.plot(action1_list)
# plt.plot(theta_list)
# # plt.hist(action2_list, bins=20)
# plt.legend(["0", "1", "2"])
# plt.show()
