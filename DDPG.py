#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/20 21:06
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/20 21:06
# @File         : DDPG.py
import gym
from stable_baselines3 import DDPG
from foo_env import DroneEnv
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# env = gym.make('foo-v0')
env = DroneEnv()

n_actions = env.action_space.shape[-1]

model = DDPG("MlpPolicy", env, action_noise=None, learning_rate=1e-3, buffer_size=pow(2, 20), batch_size=pow(2, 10),
             verbose=1, tensorboard_log="./DDPG_MultiUT_Time_tensorboard/")
model.learn(total_timesteps=410000, log_interval=10)
model.save("ddpg_MultiUT_Time")
env = model.get_env()

del model  # remove to demonstrate saving and loading

model = DDPG.load("ddpg_MultiUT_Time")

obs = env.reset()
R = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    R += rewards
    if dones == True:
        break

    env.render()
print(R)