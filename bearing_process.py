import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

test = torch.FloatTensor([1, 2, 3, 4, 5])
test = test.view((1, 5))
print(test.shape)
fnc = nn.Softmax(dim=1)
print(fnc(test))