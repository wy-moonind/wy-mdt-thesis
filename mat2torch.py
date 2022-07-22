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

def split_train_test(u1_all, u2_all, u3_all, u4_all, c1_all, c2_all, c3_all, c4_all):
    u1_train = u1_all[:-1]
    u2_train = u2_all[:-1]
    u3_train = u3_all[:-1]
    u4_train = u4_all[:-1]
    u1_test = [u1_all[-1]]
    u2_test = [u2_all[-1]]
    u3_test = [u3_all[-1]]
    u4_test = [u4_all[-1]]
    c1_train = c1_all[:-1]
    c2_train = c2_all[:-1]
    c3_train = c3_all[:-1]
    c4_train = c4_all[:-1]
    c1_test = [c1_all[-1]]
    c2_test = [c2_all[-1]]
    c3_test = [c3_all[-1]]
    c4_test = [c4_all[-1]]

    u_train = u1_train + u2_train + u3_train + u4_train
    c_train = c1_train + c2_train + c3_train + c4_train 
    u_test = u1_test + u2_test + u3_test + u4_test
    c_test = c1_test + c2_test + c3_test + c4_test

    return u_train, c_train, u_test, c_test


def case_data(path:str):
    case_all = scio.loadmat(path)
    u1 = case_all.get('u1')
    u2 = case_all.get('u2')
    u3 = case_all.get('u3')
    u4 = case_all.get('u4')
    print(u1.shape, u2.shape, u3.shape, u4.shape)

    c1 = case_all.get('c1')
    c2 = case_all.get('c2')
    c3 = case_all.get('c3')
    c4 = case_all.get('c4')

    len1 = c1.shape[0]
    len2 = c2.shape[0]
    len3 = c3.shape[0]
    len4 = c4.shape[0]
    
    u1_all, c1_all = split_data(u1, c1, len1)
    u2_all, c2_all = split_data(u2, c2, len2)
    u3_all, c3_all = split_data(u3, c3, len3)
    u4_all, c4_all = split_data(u4, c4, len4)

    print(len(u1_all), len(c1_all))

    u_train, c_train, u_test, c_test = split_train_test(u1_all, u2_all, u3_all, u4_all, c1_all, c2_all, c3_all, c4_all)

    u_train = torch.FloatTensor(u_train)
    c_train = torch.FloatTensor(c_train)

    u_test = torch.FloatTensor(u_test)
    c_test = torch.FloatTensor(c_test)

    print(u_train.shape, u_test.shape)

    torch.save(u_train, '../data/case_data/train/train_u_fd28_inner.pt')
    torch.save(c_train, '../data/case_data/train/train_y_fd28_inner.pt')
    torch.save(u_test, '../data/case_data/test/test_u_fd28_inner.pt')
    torch.save(c_test, '../data/case_data/test/test_y_fd28_inner.pt')

    pass


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



if __name__ == '__main__':
    path = '../data/case_fd28_inner.mat'
    case_data(path)
    