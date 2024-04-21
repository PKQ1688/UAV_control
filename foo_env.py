#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/4/20 20:44
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/4/20 20:44
# @File         : foo_env.py
# import gym
import casadi as ca
import gymnasium as gym
import numpy as np
# from casadi import DM
# from casadi import Opti
# from casadi import SX, vertcat, Function
# from gym import spaces
from gymnasium import spaces

import globe


class DroneEnv(gym.Env):
    """
    自定义的无人机环境，无人机在空间内移动。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, Train=True, Trajectory_mode='Kmeans', ):
        super(DroneEnv, self).__init__()
        globe._init()
        # the initial location of UAV-RIS
        globe.set_value('L_U', [0, 0, 20])  # [x, y, z]
        # the location of AP/BS
        globe.set_value('L_AP', [0, 0, 10])
        # 8 antennas for AP
        globe.set_value('BS_Z', 8)
        # 16 reflective elements for RIS
        globe.set_value('RIS_L', 16)

        if Train:
            UT_0 = np.loadtxt("CreateData/Train_Trajectory_UT_0.csv", delimiter=",")
            UT_1 = np.loadtxt("CreateData/Train_Trajectory_UT_1.csv", delimiter=",")
            UT_2 = np.loadtxt("CreateData/Train_Trajectory_UT_2.csv", delimiter=",")

            globe.set_value('UT_0', UT_0)
            globe.set_value('UT_1', UT_1)
            globe.set_value('UT_2', UT_2)

            if Trajectory_mode == 'Fermat':
                UAV_Trajectory = np.loadtxt("CreateData/Fermat_Train_Trajectory_3.csv", delimiter=",")
            else:
                UAV_Trajectory = np.loadtxt("CreateData/Kmeans_Train_Trajectory_3.csv", delimiter=",")

            globe.set_value('UAV_Trajectory', UAV_Trajectory)

        else:
            UT_0 = np.loadtxt("CreateData/Test_Trajectory_UT_0.csv", delimiter=",")
            UT_1 = np.loadtxt("CreateData/Test_Trajectory_UT_1.csv", delimiter=",")
            UT_2 = np.loadtxt("CreateData/Test_Trajectory_UT_2.csv", delimiter=",")

            globe.set_value('UT_0', UT_0)
            globe.set_value('UT_1', UT_1)
            globe.set_value('UT_2', UT_2)

            if Trajectory_mode == 'Fermat':
                UAV_Trajectory = np.loadtxt("CreateData/Fermat_Test_Trajectory_3.csv", delimiter=",")
            else:
                UAV_Trajectory = np.loadtxt("CreateData/Kmeans_Test_Trajectory_3.csv", delimiter=",")

            globe.set_value('UAV_Trajectory', UAV_Trajectory)

        # 定义动作空间和观察空间的范围
        self.action_space = spaces.Box(0, 1, shape=(20,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 20, shape=(4,), dtype=np.float32)

        # 初始化状态
        self.state = np.zeros(4 * self.n_uavs)
        self.reset()

    def step(self, action):
        # 获取当前状态
        current_state = self.state.reshape((4, self.n_uavs))

        # 对每个UAV执行MPC
        u_opt = np.zeros((1, self.n_uavs))
        for i in range(self.n_uavs):
            solver = create_mpc_controller(1)
            x0 = current_state[:, i]
            sol = solver(p=x0)
            u = sol['x'].full().flatten()[:10]  # 获取整个控制序列
            u_opt[0, i] = u[0]  # 只应用第一个控制输入

        # 更新状态（这里需要定义状态更新逻辑）
        self.state = self.next_state(u_opt)

        # 计算成本
        cost = self.calculate_cost(u_opt)
        reward = -cost
        done = False
        info = {}
        return self.state, reward, done, info

    def reset(self):
        """
        重置环境状态。
        """
        self.state = np.zeros(self.observation_space.shape[0])
        return self.state

    def render(self, mode='human', close=False):
        """
        可视化环境（可选实现）。
        """
        pass  # 对于简单演示，不实现可视化

    def calculate_cost(self, action):
        # 根据动作计算成本，此处仅为示例
        return np.random.rand()

    def next_state(self, action):
        # 计算下一个状态，此处简化处理
        return self.state + np.random.normal(0, 0.1, 4 * self.n_uavs)


# 实现动力学模型的函数
def create_mpc_controller(n_uavs, horizon=10):
    # MPC 控制器参数
    dt = 0.1  # 时间间隔
    nx = 4  # 状态空间维度
    nu = 1  # 控制输入维度

    # 声明变量
    U = ca.SX.sym('U', nu, horizon)  # 控制序列
    X = ca.SX.sym('X', nx, horizon + 1)  # 状态序列
    P = ca.SX.sym('P', nx)  # 初始状态

    # 状态转移矩阵和控制矩阵，这里需要根据实际动力学进行定义
    A = ca.DM.eye(nx)  # 简单的单位矩阵代表状态转移
    B = ca.DM.zeros(nx, nu)  # 控制输入影响矩阵
    B[2, 0] = dt
    B[3, 0] = dt

    # 成本函数和约束
    Q = ca.DM.eye(nx)  # 状态权重矩阵
    R = ca.DM.eye(nu)  # 控制权重矩阵
    obj = 0  # 目标函数
    g = []  # 约束条件

    # 构建预测模型和目标函数
    st = P
    for k in range(horizon):
        st_next = A @ st + B @ U[:, k]
        obj += (st - X[:, k]).T @ Q @ (st - X[:, k]) + U[:, k].T @ R @ U[:, k]
        st = st_next
        g.append(st_next - X[:, k + 1])

    # 创建优化问题
    opts = {'verbose': False, 'ipopt.print_level': 0}
    nlp = {'x': ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1)),
           'f': obj,
           'g': ca.vertcat(*g),
           'p': P}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    return solver
