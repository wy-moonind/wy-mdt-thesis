import numpy as np
import torch
from torch import nn
from loss_func import AvgLoss

import matplotlib.pyplot as plt

from utils import import_data


class PassFilter(nn.Module):

    def __init__(self, seq_len, order, critical_freq: np.ndarray, sampling_rate=12000, mode=1):
        super(PassFilter, self).__init__()

        self.critical_freqs = critical_freq
        self.sampling_rate = sampling_rate
        self.order = order
        # self.hidden_param_size = self.get_num_weights()
        self.sampling_period = 1 / sampling_rate  # second
        self.mode = mode  # 1 for low-pass, 2 for band-pass, 3 for high-pass
        self.seq_len = seq_len
        # self.hidden_state = torch.tensor([[0], [0]], dtype=float)
        # self.hidden_state_vec = torch.tensor([[0], [0]], dtype=float)

        # self.weight = nn.Parameter(torch.Tensor(np.ones((8, 1))))
        self.weight_a = nn.Parameter(torch.tensor(np.ones((self.order, self.order))))
        self.weight_b = nn.Parameter(torch.tensor(np.ones((self.order, 1))))
        self.weight_c = nn.Parameter(torch.tensor(np.ones((1, self.order))))
        # self.on_gpu = torch.cuda.is_available()
        # # transfer to gpu
        # if self.on_gpu:
        #     self.transfer_to_gpu()
        # initialize parameters
        self.reset_parameter()
        # self.const_reset_parameter()

    def reset_parameter(self):
        # print("reset param")
        for weight in self.parameters():
            nn.init.uniform_(weight, -1, 1)

    def const_reset_parameter(self):
        for weight in self.parameters():
            nn.init.constant_(weight, 1.0)

    # def transfer_to_gpu(self):
    #     self.hidden_state = self.hidden_state.cuda()
    #     self.hidden_state_vec = self.hidden_state_vec.cuda()

    def detach_hidden_(self):
        self.hidden_state.requires_grad = False
        self.hidden_state._grad_fn_ = None

    def forward(self, input):
        # torch.reshape(input, [self.seq_len, 1])
        input = input.view(1, self.seq_len)
        assert input.shape[0] or input.shape[1] == self.seq_len
        y = torch.zeros((1, self.seq_len))
        eye = torch.eye(self.order)
        hidden = torch.tensor([[input[0, 0]], [0]], dtype=float)
        # if self.on_gpu:
        #     y = y.cuda()
        #     eye = eye.cuda()
        # discrete
        a_k = torch.exp(self.weight_a * self.sampling_period)
        b_k = torch.mm(eye / self.weight_a, torch.mm((a_k - eye), self.weight_b))
        print("a_k: ", a_k, self.weight_a, "\nb_k: ", b_k, self.weight_b)
        # do sequence processing
        for i in range(self.seq_len):
            # x_t1 = torch.mm((self.weight_a * self.sampling_period + eye), hidden) + (self.sampling_period * self.weight_b * input[0, i])
            # x_t1 = torch.mm(self.weight_a, hidden) + input[0, i] * self.weight_b
            x_t1 = torch.mm(a_k, hidden) + b_k * input[0, i]
            y_t = torch.mm(self.weight_c, hidden)  # hidden was the last-step state
            hidden = x_t1
            y[0, i] = y_t
        assert y.shape == input.shape
        return y

    def predict(self, input):
        return self.forward(input).detach()


def normalization(data):
    return (data - min(data)) / (max(data) - min(data))

def hidden_forward(self, u, y_init):
        self.seq_len = u.shape[1]
        y = torch.zeros((1, self.seq_len), dtype=float)
        noise1 = torch.FloatTensor(3, 1)
        noise2 = torch.FloatTensor(1)
        u = u.float()
        y_init = y_init.float()
        hidden = torch.FloatTensor(self.order, 1)
        u = u.view(1, self.seq_len)
        nn.init.constant_(hidden, 0)
        for i in range(self.seq_len):
            # nn.init.uniform_(noise1, 0, 0.1)
            # nn.init.uniform_(noise2, 0, 0.1)
            x_t1 = torch.mm(self.weight_a, hidden) + self.weight_b * u[:, i]
            y_t = torch.mm(self.weight_c, hidden) + self.weight_d * u[:, i]
            # if i == 0:
            #     x_t1 += self.weight_l1 * y_init
            #     y_t += y_init * self.weight_l2
            hidden = x_t1
            y[0, i] = y_t
        return y


def main():
    testdata = np.load("./new_data/train/lowpass_train_2.npy")
    x_train, x_test, y_train, y_test = import_data("./dataset/")

    # print(x_train[0].shape)

    neuron = PassFilter(576, 2, np.array([2000, 4000]))
    y = neuron.forward(torch.tensor(x_test[0].T))
    # print(y, y.shape)
    plt.figure(0)
    # plt.plot(y)
    plt.plot(x_train[0])
    plt.plot(testdata[0, :])
    plt.legend(["neuron_output", "original", "filtered"])
    # plt.ylim(-1, 1)

    plt.show()


if __name__ == "__main__":
    main()
