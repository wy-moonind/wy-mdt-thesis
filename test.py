import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import sys
import json
import copy

from engine import ParallelModel

# points = torch.linspace(-3, 3, 100)
# tanh = nn.Tanh()
# relu = nn.ReLU()
# softplus = nn.Softplus()
# sigmoid = nn.Sigmoid()

# tanh_ans = tanh(points)
# relu_ans = relu(points)
# softplus_ans = softplus(points)
# relu_ans_md = relu(points) - 1
# softplus_ans_md = softplus(points) - 1
# sigmoid_ans = sigmoid(points)

# plt.figure(0)  # tanh result
# plt.plot(points, softplus_ans, linewidth=3, color='g')
# plt.plot(points, softplus_ans_md, linewidth=3, color='b')
# # plt.ylim(-2, 2)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -2, 2, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# # plt.title('Tanh activation function')
# plt.legend(["SoftPlus activation", "Shifted SoftPlus activation"])
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/SoftPlus_all', dpi=300)

# plt.figure(1)  # tanh result
# plt.plot(points, relu_ans, linewidth=3, color='g')
# plt.ylim(-4, 4)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -4, 4, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# # plt.title('ReLU activation function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/relu_og', dpi=300)

# plt.figure(2)  # tanh result
# plt.plot(points, softplus_ans, linewidth=3, color='g')
# plt.ylim(-4, 4)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -4, 4, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# # plt.title('SoftPlus activation function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/softplus_og', dpi=300)

# plt.show()
# data_inner07 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/inner_fd07_preprocessed.mat")
# inner07 = data_inner07.get("inner_fd07_4")
# data_inner14 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/inner_fd14_preprocessed.mat")
# inner14 = data_inner14.get("inner_fd14_4")
# data_inner21 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/inner_fd21_preprocessed.mat")
# inner21 = data_inner21.get("inner_fd21_4")
# data_inner28 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/inner_fd28_preprocessed.mat")
# inner28 = data_inner28.get("inner_fd28_4")

# time = np.linspace(0, 0.5, 3000)

# plt.figure(figsize = (10,8))
# plt.subplot(4, 1, 1)
# plt.plot(time, inner07[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Inner race, fd=0.007"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(4, 1, 2)
# plt.plot(time, inner14[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Inner race, fd=0.014"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(4, 1, 3)
# plt.plot(time, inner21[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Inner race, fd=0.021"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(4, 1, 4)
# plt.plot(time, inner21[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Inner race, fd=0.028"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)

# plt.savefig('../figs/final_figs/cwru_inner_example', dpi=300, bbox_inches='tight')
# plt.show()

# data_ball07 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/ball_fd07_preprocessed.mat")
# ball07 = data_ball07.get("ball_fd07_4")
# data_ball14 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/ball_fd14_preprocessed.mat")
# ball14 = data_ball14.get("ball_fd14_4")
# data_ball21 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/ball_fd21_preprocessed.mat")
# ball21 = data_ball21.get("ball_fd21_4")

# time = np.linspace(0, 0.5, 3000)

# plt.figure(figsize = (10,6))
# plt.subplot(3, 1, 1)
# plt.plot(time, ball07[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Ball, fd=0.007"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(3, 1, 2)
# plt.plot(time, ball14[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Ball, fd=0.014"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(3, 1, 3)
# plt.plot(time, ball21[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Ball, fd=0.021"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.tight_layout()
# plt.savefig('../figs/final_figs/cwru_ball_example', dpi=300, bbox_inches='tight')
# plt.show()

# data_outer07 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/outer_fd07_preprocessed.mat")
# outer07 = data_outer07.get("outer_fd07_4")
# data_outer14 = scio.loadmat("D:/desktop/Thesis_new/bearing_data/case/preprocessed/outer_fd21_preprocessed.mat")
# outer14 = data_outer14.get("outer_fd21_4")

# time = np.linspace(0, 0.5, 3000)

# plt.figure(figsize = (10,4))
# plt.subplot(2, 1, 1)
# plt.plot(time, outer07[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Outer race, fd=0.007"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.subplot(2, 1, 2)
# plt.plot(time, outer14[0:3000, :])
# plt.xlabel("Time[s]", fontsize=12)
# plt.ylabel("Acceleration $[m/s^2]$", fontsize=12)
# plt.legend(["Outer race, fd=0.021"], loc="upper right", fontsize=12)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.tight_layout()
# plt.savefig('../figs/final_figs/cwru_outer_example', dpi=300, bbox_inches='tight')
# plt.show()



