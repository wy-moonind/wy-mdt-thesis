import torch
from torch import nn
import numpy as np
from utils import import_data
from torch import fft
import scipy.io as scio
import torch.utils.data as Data

import matplotlib.pyplot as plt
import sys

class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope
 
def mapminmax(x, ymin=-1, ymax=+1):
	x = np.asanyarray(x)
	xmax = x.max(axis=-1)
	xmin = x.min(axis=-1)
	if (xmax==xmin).any():
		raise ValueError("some rows have no variation")
	slope = ((ymax-ymin) / (xmax - xmin))
	intercept = (-xmin*(ymax-ymin)/(xmax-xmin)) + ymin
	ps = MapMinMaxApplier(slope, intercept)
	return ps(x)

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
    u1_train = u1_all[0:150]
    u2_train = u2_all[0:150]
    u3_train = u3_all[0:150]
    u4_train = u4_all[0:150]
    u1_test = u1_all[150:]
    u2_test = u2_all[150:]
    u3_test = u3_all[150:]
    u4_test = u4_all[150:]
    c1_train = c1_all[0:150]
    c2_train = c2_all[0:150]
    c3_train = c3_all[0:150]
    c4_train = c4_all[0:150]
    c1_test = c1_all[150:]
    c2_test = c2_all[150:]
    c3_test = c3_all[150:]
    c4_test = c4_all[150:]
    c1_show = [c1_test[0]]
    c2_show = [c2_test[0]]
    c3_show = [c3_test[0]]
    c4_show = [c4_test[0]]
    u1_show = [u1_test[0]]
    u2_show = [u2_test[0]]
    u3_show = [u3_test[0]]
    u4_show = [u4_test[0]]
    c_show = c1_show + c2_show + c3_show + c4_show
    u_show = u1_show + u2_show + u3_show + u4_show
    print(len(c_show), len(u_show))

    u_train = u1_train + u2_train + u3_train + u4_train
    c_train = c1_train + c2_train + c3_train + c4_train 
    u_test = u1_test + u2_test + u3_test + u4_test
    c_test = c1_test + c2_test + c3_test + c4_test

    return u_train, c_train, u_test, c_test, u_show, c_show


def case_data(path:str, name:str):
    case_all = scio.loadmat(path)
    u1 = case_all.get('u1')
    u2 = case_all.get('u2')
    u3 = case_all.get('u3')
    u4 = case_all.get('u4')
    # print(u1.shape, u2.shape, u3.shape, u4.shape)

    c1 = case_all.get('c1')
    c2 = case_all.get('c2')
    c3 = case_all.get('c3')
    c4 = case_all.get('c4')

    len1 = c1.shape[0]
    len2 = c2.shape[0]
    len3 = c3.shape[0]
    len4 = c4.shape[0]
    # print(len1)
    
    u1_all, c1_all = split_data(u1, c1, len1)
    u2_all, c2_all = split_data(u2, c2, len2)
    u3_all, c3_all = split_data(u3, c3, len3)
    u4_all, c4_all = split_data(u4, c4, len4)

    u_train, c_train, u_test, c_test, u_show, c_show = split_train_test(u1_all, u2_all, u3_all, u4_all, c1_all, c2_all, c3_all, c4_all)
    # print(len(u_train), len(u_test))

    u_train = torch.FloatTensor(u_train)
    c_train = torch.FloatTensor(c_train)

    u_test = torch.FloatTensor(u_test)
    c_test = torch.FloatTensor(c_test)

    u_show = torch.FloatTensor(u_show)
    c_show = torch.FloatTensor(c_show)

    print(name)
    print(u_train.shape, u_test.shape)

    torch.save(u_train, '../data/case_data/train/train_u_' + name + '.pt')
    torch.save(c_train, '../data/case_data/train/train_y_' + name + '.pt')
    torch.save(u_test, '../data/case_data/test/test_u_' + name + '.pt')
    torch.save(c_test, '../data/case_data/test/test_y_' + name + '.pt')
    torch.save(u_show, '../data/case_data/test/show_u_' + name + '.pt')
    torch.save(c_show, '../data/case_data/test/show_y_' + name + '.pt')

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


def femto_main():
    data = scio.loadmat('../data/final_femto.mat')
    u1 = data.get('u1')
    u2 = data.get('u2')
    u3 = data.get('u3')
    u1_test = data.get('u1_test')
    u2_test = data.get('u2_test')
    u3_test = data.get('u3_test')

    d1 = data.get('d1')
    d2 = data.get('d2')
    d3 = data.get('d3')
    d1_test = data.get('d1_test')
    d2_test = data.get('d2_test')
    d3_test = data.get('d3_test')

    print(u1.shape, d1.shape)
    len1 = d1.shape[0]
    len2 = d2.shape[0]
    len3 = d3.shape[0]

    u1_all, d1_all = split_data(u1, d1, len1)
    u1_test_all, d1_test_all = split_data(u1_test, d1_test, 50)
    u1_show = [u1_test_all[1]]
    d1_show = [d1_test_all[1]]
    u2_all, d2_all = split_data(u2, d2, len2)
    u2_test_all, d2_test_all = split_data(u2_test, d2_test, 50)
    u2_show = [u2_test_all[1]]
    d2_show = [d2_test_all[1]]
    u3_all, d3_all = split_data(u3, d3, len3)
    u3_test_all, d3_test_all = split_data(u3_test, d3_test, 50)
    u3_show = [u3_test_all[1]]
    d3_show = [d3_test_all[1]]
    u_show = u1_show + u2_show + u3_show
    y_show = d1_show + d2_show + d3_show
    print(len(u_show), len(y_show))
    print(u_show[1].shape, y_show[1].shape)

    u_train = u1_all + u2_all + u3_all
    y_train = d1_all + d2_all + d3_all
    u_test = u1_test_all + u2_test_all + u3_test_all
    y_test = d1_test_all + d2_test_all + d3_test_all

    torch.save(torch.FloatTensor(u_train), '../data/femto_data/train/train_u_all.pt')
    torch.save(torch.FloatTensor(y_train), '../data/femto_data/train/train_y_all.pt')
    torch.save(torch.FloatTensor(u_test), '../data/femto_data/test/test_u_all.pt')
    torch.save(torch.FloatTensor(y_test), '../data/femto_data/test/test_y_all.pt')
    torch.save(torch.FloatTensor(u_show), '../data/femto_data/test/show_u_all.pt')
    torch.save(torch.FloatTensor(y_show), '../data/femto_data/test/show_y_all.pt')


if __name__ == '__main__':
    # femto_main()
    case_data('../data/final_ball21.mat', 'ball21')
    