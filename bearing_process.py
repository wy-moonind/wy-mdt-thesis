import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import sys

# points = torch.linspace(-3, 3, 100)
# tanh = nn.Tanh()
# relu = nn.ReLU()
# softplus = nn.Softplus()
#
# tanh_ans = tanh(points) * 1.5
# relu_ans = relu(points) - 1.5
# softplus_ans = softplus(points) - 1.5
#
# plt.figure(0)  # tanh result
# plt.plot(points, tanh_ans)
# plt.ylim(-2, 2)
# plt.xlim(-4, 4)
# plt.grid(True)
# plt.title('')
#
# plt.figure(1)  # tanh result
# plt.plot(points, relu_ans)
# plt.ylim(-4, 4)
# plt.xlim(-4, 4)
# plt.grid(True)
#
# plt.figure(2)  # tanh result
# plt.plot(points, softplus_ans)
# plt.ylim(-4, 4)
# plt.xlim(-4, 4)
# plt.grid(True)

# plt.show()

order_str = sys.argv[1]
order = int(order_str)
print(order)