import scipy.io as scio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn

from engine import StateModel
from validation import validation
from data import MyData
from loss_func import RMSELoss, r2_loss

def woobs_figs(data: str):
    data_gen = MyData()
    criterion = RMSELoss()

    train_set, test_set, show_set = data_gen.get_case_data(data)
    print(len(show_set))
    outer07_model = torch.load('../models/outer07_15_1_serial_woobs.pt')

    dump3, dump4, dump5, dump4 = validation(
        outer07_model, show_set, criterion, num_data=4, origin=False, obs=False, show=True, fig_num=1) 

    # plt.savefig(fig_path + name + '_val', dpi=300, bbox_inches='tight')
    plt.show()

def activation_figs():
    points = torch.linspace(-3, 3, 100)
    tanh = nn.Tanh()
    relu = nn.ReLU()
    softplus = nn.Softplus()
    sigmoid = nn.Sigmoid()

    tanh_ans = tanh(points)
    relu_ans = relu(points)
    softplus_ans = softplus(points)
    relu_ans_md = relu(points) - 1
    softplus_ans_md = softplus(points) - 1
    sigmoid_ans = sigmoid(points)

    plt.figure(0)  # tanh result
    plt.plot(points, tanh_ans, linewidth=3, color='g')
    plt.plot(points, sigmoid_ans, linewidth=3, color='b')
    # plt.ylim(-2, 2)
    plt.xlim(-3.5, 3.5)
    plt.vlines(0, -2, 2, colors='k')
    plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
    plt.grid(True)
    # plt.title('Tanh activation function')
    plt.legend(["Tanh activation", "Sigmoid activation"], fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('f(x)', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig('../figs/tanh_sigmoid', dpi=300, bbox_inches='tight')

    plt.figure(1)  # relu result
    plt.plot(points, relu_ans, linewidth=3, color='g')
    plt.plot(points, relu_ans_md, linewidth=3, color='b')
    plt.ylim(-4, 4)
    plt.xlim(-3.5, 3.5)
    plt.vlines(0, -4, 4, colors='k')
    plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
    plt.grid(True)
    # plt.title('ReLU activation function')
    plt.legend(["ReLU activation", "Shifted ReLU activation"], fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('f(x)', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig('../figs/relu_all', dpi=300, bbox_inches='tight')

    plt.figure(2)  # softplus result
    plt.plot(points, softplus_ans, linewidth=3, color='g')
    plt.plot(points, softplus_ans_md, linewidth=3, color='b')
    plt.ylim(-4, 4)
    plt.xlim(-3.5, 3.5)
    plt.vlines(0, -4, 4, colors='k')
    plt.hlines(0, -3.5, 3.5, linestyles='dashed', colors='k')
    plt.grid(True)
    # plt.title('SoftPlus activation function')
    plt.legend(["SoftPlus activation", "Shifted SoftPlus activation"], fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('f(x)', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig('../figs/softplus_all', dpi=300, bbox_inches='tight')

    plt.show()




if __name__ == "__main__":
    woobs_figs('outer07')
    # activation_figs()