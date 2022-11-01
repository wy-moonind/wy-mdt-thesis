from math import comb
from turtle import forward
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
import os
import json


class ParallelModel(nn.Module):

    def __init__(self, order, in_dim=1, out_dim=1,layers=2, seq_len=112, activation='Tanh', device=torch.device('cpu')):
        super(ParallelModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.layers = layers
        self.observer = True
        self.activation = activation
        self.acti = nn.Tanh()
        assert self.layers >= 2
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=self.in_dim,
                                        out_dim=self.out_dim,
                                        observer=True,
                                        activation=self.activation,
                                        device=device)
        if self.layers >= 2:                                    
            self.state_layer2 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device)
        if self.layers >= 3:                                    
            self.state_layer3 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device)
        if self.layers >= 4:                                    
            self.state_layer4 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device)
        if self.layers >= 5:                                    
            self.state_layer5 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device)
        if self.layers >= 6:                                    
            self.state_layer5 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device)
        self.fully_connected = nn.Linear(
            in_features=self.seq_len*self.layers, out_features=self.seq_len, bias=False).to(device)
        for weight in self.fully_connected.parameters():
            nn.init.uniform_(weight, 0, 0.5)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None):
        assert y_obs.shape[1] == self.seq_len
        out1 = self.state_layer1(x, y_obs=y_obs)
        all = [out1]
        if self.layers >= 2: 
            out2 = self.state_layer2(x, y_obs=y_obs)
            all.append(out2)
        if self.layers >= 3: 
            out3 = self.state_layer2(x, y_obs=y_obs)
            all.append(out3)
        if self.layers >= 4: 
            out4 = self.state_layer2(x, y_obs=y_obs)
            all.append(out4)
        if self.layers >= 5: 
            out5 = self.state_layer2(x, y_obs=y_obs)
            all.append(out5)
        if self.layers >= 6: 
            out6 = self.state_layer2(x, y_obs=y_obs)
            all.append(out6)
        combined = torch.cat(all, 1)
        out = self.fully_connected(combined)

        return self.acti(out)


class StateModel(nn.Module):

    def __init__(self, order, in_dim=1, out_dim=1, layers=1, observer=False, activation='Tanh', device=torch.device('cpu')):
        super(StateModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.observer = observer
        self.activation = activation
        if layers>=2:
            self.state_space1 = StateSpace(2, in_dim=self.in_dim, activation='None', device=device)
        if layers>=3:
            self.state_space2 = StateSpace(4, in_dim=2, activation='None', device=device)
        if layers>=4:
            self.state_space3 = StateSpace(6, in_dim=4, activation='None', device=device)
        if layers>=5:
            self.state_space4 = StateSpace(8, in_dim=6, activation='None', device=device)
        if layers>=6:
            self.state_space5 = StateSpace(10, in_dim=8, activation='None', device=device)
        last_in_dim = in_dim if layers==1 else 2*layers-2
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=last_in_dim,
                                        out_dim=self.out_dim,
                                        observer=True,
                                        activation=self.activation,
                                        device=device)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None):
        if y_obs is not None:
            assert self.observer
        if self.layers >=2:
            out = self.state_space1(x)
        if self.layers >=3:
            out = self.state_space2(out)
        if self.layers >=4:
            out = self.state_space3(out)
        if self.layers >=5:
            out = self.state_space4(out)
        if self.layers >=6:
            out = self.state_space5(out)
        out = self.state_layer1(out if self.layers>1 else x, y_obs=y_obs)
        return out


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
    layer = config['layer']
    structure = config['structure']
    note = config['note']
    data = config['data']
    fig_path = config['fig_path']
    name = data + '_' + str(order) + '_' + str(layer) + '_' + structure + '_' + note
    print(name)

    # start with main function
    if structure == 'serial':
        model = StateModel(order, in_dim=2, out_dim=1, layers=layer,
                           observer=True, activation='None', device=device)
    elif structure == 'parallel':
        model = ParallelModel(order, in_dim=2, out_dim=1, layers=layer,
                              seq_len=112, activation="None", device=device)
    else:
        print('Unrecognized model structure:', structure)
        sys.exit(-2)

    # for name, param in model.named_parameters():
    #     print(name, param.size())

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
                          learning_rate=0.0005,
                          grad_clip=30)

    # model_name = '../models/test/' + name + '.pt'
    # torch.save(model, model_name)
    print('Name of the model: ', name)
    is_good = input("Continue validation? 1 to continue, 2 abort: ")
    is_good = int(is_good)
    if is_good == 2:
        sys.exit(0)
    # plot training curve
    train_fig = plt.figure(0)
    plt.plot(train_history.train_loss)
    plt.plot(train_history.train_r2)
    plt.ylim(0, 1.5)
    plt.legend(['Training loss', 'Training R2-value'])
    plt.xlabel('Epoch')
    plt.title('Training Process')
    plt.grid(True)
    if os.path.exists(fig_path + name + '_loss.png'):
        print('found old image, removing')
        os.remove(fig_path + name + '_loss.png')
    train_fig.savefig(fig_path + name + '_loss', dpi=300)
    plt.close()

    # evaluation
    val_loss_obs, val_r2_obs, val_loss_wo, val_r2_wo = validation(
        model, test_set, criterion, num_data=4, origin=False, obs=True, show=True, fig_num=1)
    print('validation loss with obs = ', val_loss_obs,
          '\nR2 loss with obs = ', val_r2_obs)
    print('validation loss wo obs = ', val_loss_wo,
          '\nR2 loss wo obs = ', val_r2_wo)
    print(config)
    if os.path.exists(fig_path + name + '_val.png'):
        print('found old image, removing')
        os.remove(fig_path + name + '_val.png')
    plt.savefig(fig_path + name + '_val', dpi=300)
    plt.close()
    # plt.show()

    return 0


if __name__ == '__main__':
    main()
