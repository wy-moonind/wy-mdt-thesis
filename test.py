import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import sys
import json

from engine import ParallelModel

# points = torch.linspace(-3, 3, 100)
# tanh = nn.Tanh()
# relu = nn.ReLU()
# softplus = nn.Softplus()

# tanh_ans = tanh(points) * 1.5
# relu_ans = relu(points)
# softplus_ans = softplus(points)

# plt.figure(0)  # tanh result
# plt.plot(points, tanh_ans, linewidth=3, color='g')
# plt.ylim(-2, 2)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -2, 2, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# plt.title('Tanh activation function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/tanh', dpi=300)

# plt.figure(1)  # tanh result
# plt.plot(points, relu_ans, linewidth=3, color='g')
# plt.ylim(-4, 4)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -4, 4, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# plt.title('ReLU activation function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/relu_org', dpi=300)

# plt.figure(2)  # tanh result
# plt.plot(points, softplus_ans, linewidth=3, color='g')
# plt.ylim(-4, 4)
# plt.xlim(-3.5, 3.5)
# plt.vlines(0, -4, 4, colors='k')
# plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
# plt.grid(True)
# plt.title('SoftPlus activation function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.savefig('../figs/activation/softplus_org', dpi=300)

# plt.show()

# with open('./config.json') as f:
#     config = json.load(f)

# order = config['order']
# name = config['name']
# data = config['data']
# fig_path = config['fig_path']

# print(config)

model = ParallelModel(5, in_dim=2, out_dim=1)
