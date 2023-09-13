from SheepDogEnv import SheepDogEnv
import numpy as np

# while True:
for i in range(2000):
    R = np.random.randint(200, 400)
    d_V = np.random.randint(int(R/6), int(R/3))
    L1 = np.pi/(np.pi+1)*R
    s_V = np.random.randint(int(L1/(np.pi*R)*d_V)+1, d_V)
    env = SheepDogEnv(circle_R=R, sheep_v=s_V, dog_v=d_V,
                      sec_split_n=10, store_mode=True, render_mode=False)
    action = env.action_space.sample()[0]
    observation, reward, done, _, info = env.step(action)  # 和环境交互
    env.save(
        save_path="C:/Users/labadmin/Desktop/Python/VSCode/sheep-dog-rl/1step_store/1"
    )
