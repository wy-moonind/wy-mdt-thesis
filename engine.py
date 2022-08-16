from neurons import StateNeuron, StateSpace
from validation import validation
from train import train
from loss_func import RMSELoss, r2_loss
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from data import MyData
import sys
import json


class StateModel(nn.Module):

    def __init__(self, order, in_dim=1, out_dim=1, observer=False, activation='Tanh', device=torch.device('cpu')):
        super(StateModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.observer = observer
        self.activation = activation
        self.state_space1 = StateSpace(2, in_dim=self.in_dim, activation=self.activation, device=device)
        self.state_space2 = StateSpace(4, in_dim=2, activation=self.activation, device=device)
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=4,
                                        out_dim=self.out_dim,
                                        observer=True,
                                        activation=self.activation,
                                        device=device)
        # self.state_layer2 = StateNeuron(self.order,
        #                                 in_dim=2,
        #                                 out_dim=4,
        #                                 observer=False,
        #                                 device=device)
        # self.state_layer3 = StateNeuron(self.order,
        #                                 in_dim=4,
        #                                 out_dim=self.out_dim,
        #                                 observer=True,
        #                                 device=device)
        # self.state_layer4 = StateNeuron(self.order,
        #                                 in_dim=8,
        #                                 out_dim=10,
        #                                 observer=False,
        #                                 device=device)
        # self.state_layer5 = StateNeuron(self.order,
        #                                 in_dim=10,
        #                                 out_dim=self.out_dim,
        #                                 observer=True,
        #                                 device=device)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None):
        if y_obs is not None:
            assert self.observer
        out = self.state_space1(x)
        out1 = self.state_space2(out)
        out2 = self.state_layer1(out1, y_obs=y_obs)
        # out2 = self.state_layer2(out1)
        # out3 = self.state_layer3(out2, y_obs=y_obs)
        # out4 = self.state_layer4(out3)
        # out5 = self.state_layer5(out4, y_obs=y_obs)
        return out2


# note for model name format:
# racefd_order_layer_type
# type: obs, nlobs, none

def main():
    EPOCH = 250
    BATCH_SIZE = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert len(sys.argv) == 2
    with open('./config.json') as f:
        config = json.load(f)

    order = config['order']
    name = config['name']
    data = config['data']
    fig_path = config['fig_path']

    # start with main function
    model = StateModel(order, in_dim=2, out_dim=1,
                       observer=True, activation='None', device=device)
    print('Number of parameters: ', model.count_parameters())
    criterion = RMSELoss()

    data_gen = MyData()
    train_set, test_set, train_size = data_gen.get_case_data(data)
    print("Dataset size: ", train_size)

    # training
    train_history = train(model,
                          criterion,
                          r2_loss,
                          epoch=EPOCH,
                          train_set=train_set,
                          batch_size=BATCH_SIZE,
                          optimizer='Adam',
                          learning_rate=0.001,
                          grad_clip=30)

    # model_name = '../models/test/' + name + '.pt'
    # torch.save(model, model_name)
    # plot training curve
    plt.figure(0)
    plt.plot(train_history.train_loss)
    plt.plot(train_history.train_r2)
    plt.ylim(0, 1)
    plt.legend(['Training loss', 'Training R2-value'])
    plt.xlabel('Epoch')
    plt.title('Training Process')
    plt.grid(True)
    plt.savefig(fig_path + name + '_loss', dpi=300)

    # evaluation
    val_loss_obs, val_r2_obs, val_loss_wo, val_r2_wo = validation(
        model, test_set, criterion, num_data=4, origin=False, obs=True, show=True, fig_num=1)
    print('validation loss with obs = ', val_loss_obs,
          '\nR2 loss with obs = ', val_r2_obs)
    print('validation loss wo obs = ', val_loss_wo,
          '\nR2 loss wo obs = ', val_r2_wo)
    print(config)
    plt.savefig(fig_path + name + '_val', dpi=300)

    plt.show()

    return 0


if __name__ == '__main__':
    main()
