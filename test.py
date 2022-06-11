import torch
from torch import nn
import numpy as np
from utils import import_data
from torch import fft
import scipy.io as scio
import torch.utils.data as Data

import matplotlib.pyplot as plt
import sys


def xjtu_data():
    u_dict = scio.loadmat('./bearing_data/xjtu/u_outer_all.mat')
    y_dict = scio.loadmat('./bearing_data/xjtu/y_outer_all.mat')

    u_outer_1_1 = u_dict.get('u_outer_1_1')
    y_outer_1_1 = y_dict.get('mat_acc_x_outer_1_1')

    u_outer_2_2 = u_dict.get('u_outer_2_2')
    y_outer_2_2 = y_dict.get('mat_acc_x_outer_2_2')

    # sys.exit()

    y_outer_1_1 = y_outer_1_1[:, ::3]
    u_outer_1_1 = u_outer_1_1[:, ::3, :]
    u_outer_1_1_all = []
    y_outer_1_1_all = []

    u_outer_2_2 = u_outer_2_2[:, ::3, :]
    y_outer_2_2 = y_outer_2_2[:, ::3]
    u_outer_2_2_all = []
    y_outer_2_2_all = []

    print(y_outer_1_1.shape, y_outer_2_2.shape)

    plt.figure(0)
    plt.subplot(5, 1, 1)
    plt.plot(y_outer_1_1[0, :])
    plt.subplot(5, 1, 2)
    plt.plot(y_outer_1_1[1, :])
    plt.subplot(5, 1, 3)
    plt.plot(y_outer_1_1[2, :])
    plt.subplot(5, 1, 4)
    plt.plot(y_outer_1_1[3, :])
    plt.subplot(5, 1, 5)
    plt.plot(y_outer_1_1[4, :])

    # sys.exit()

    for i in range(10):
        u_outer_1_1_all.append(u_outer_1_1[:, :, i])
        u_outer_2_2_all.append(u_outer_2_2[:, :, i])
        y_outer_1_1_all.append(y_outer_1_1[i, :])
        y_outer_2_2_all.append(y_outer_2_2[i, :])

    for i in range(10):
        y_outer_1_1_all[i] = np.hstack((y_outer_1_1_all[i], y_outer_1_1_all[i][:1]))
        y_outer_2_2_all[i] = np.hstack((y_outer_2_2_all[i], y_outer_2_2_all[i][:6]))
        while u_outer_1_1_all[i].shape[1] < 80:
            u_outer_1_1_all[i] = np.hstack((u_outer_1_1_all[i], np.array([[35], [12]])))
        while u_outer_2_2_all[i].shape[1] < 80:
            u_outer_2_2_all[i] = np.hstack((u_outer_2_2_all[i], np.array([[37.5], [11]])))

    print(y_outer_1_1_all[0].shape, y_outer_2_2_all[0].shape)

    u_outer = u_outer_1_1_all + u_outer_2_2_all
    y_outer = y_outer_1_1_all + y_outer_2_2_all

    u_outer = torch.FloatTensor(u_outer)
    y_outer = torch.FloatTensor(y_outer)

    torch.save(u_outer, './bearing_data/xjtu/u_outer.pt')
    torch.save(y_outer, './bearing_data/xjtu/y_outer.pt')

    print(u_outer.shape, y_outer.shape)

    dataset = Data.TensorDataset(u_outer, y_outer)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    # plt.figure(1)
    #
    # for idx, (batch_x, batch_y) in enumerate(loader):
    #     if idx >= 5:
    #         plt.subplot(5, 1, idx - 4)
    #         plt.plot(batch_y.numpy().T)

    plt.show()

    return None


def split_data(u, y, length):
    u_target, y_target = [], []
    for i in range(length):
        u_target.append(u[:, :, i])
        y_target.append(y[i, :])
    return u_target, y_target


def case_data():
    case_all = scio.loadmat('./bearing_data/case_fd07_outer.mat')
    u144 = case_all.get('u_144')
    u145 = case_all.get('u_145')
    u146 = case_all.get('u_146')
    u147 = case_all.get('u_147')

    y144 = case_all.get('x144')
    y145 = case_all.get('x145')
    y146 = case_all.get('x146')
    y147 = case_all.get('x147')

    u144_all, y144_all = split_data(u144, y144, 9)
    u145_all, y145_all = split_data(u145, y145, 8)
    u146_all, y146_all = split_data(u146, y146, 6)
    u147_all, y147_all = split_data(u147, y147, 6)

    print(u144_all[0].shape, y144_all[0].shape)

    u_case_all = u144_all + u145_all + u146_all + u147_all
    y_case_all = y144_all + y145_all + y146_all + y147_all

    u_case_all = torch.FloatTensor(u_case_all)
    y_case_all = torch.FloatTensor(y_case_all)

    print(u_case_all.shape, y_case_all.shape)

    torch.save(u_case_all, './bearing_data/case_fd07_u.pt')
    torch.save(y_case_all, './bearing_data/case_fd07_y.pt')

    return None


def plot_sth():
    case_all = scio.loadmat('./bearing_data/case_fd21_outer.mat')
    y246 = case_all.get('x246')
    y = y246[0, :]
    t = np.linspace(0, 1, num=y.shape[0]) / 12000

    plt.figure(0)
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration')
    plt.grid(True)
    plt.ticklabel_format(style='plain')

    plt.figure(1)
    envelope = scio.loadmat('../bearing_data/case/envelope.mat')
    es = envelope.get('es')
    f = envelope.get('f')
    plt.plot(f, es)
    plt.vlines(107.5, 0, 0.07, colors='y', linestyles='dashed')
    plt.vlines(107.5 * 2, 0, 0.07, colors='y', linestyles='dashed')
    plt.vlines(107.5 * 3, 0, 0.07, colors='y', linestyles='dashed')
    plt.vlines(107.5 * 4, 0, 0.07, colors='y', linestyles='dashed')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Envelope spectrum of outer race data')
    plt.xlim(0, 500)
    plt.ylim(0, 0.07)
    plt.grid(True)

    plt.show()

class rdm:

    def __init__(self):
        self.activ = nn.Tanh


if __name__ == '__main__':
    # case_data()
    tanh = nn.Tanh()
    x = torch.FloatTensor(2,2)
    y = tanh(x)
    print(x, y)