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
import datetime
import copy


class ParallelModel(nn.Module):
    """
    Definition of the Parallel DSSO-NN
    """

    def __init__(self, order, layers=2,
                 in_dim=1, out_dim=1,
                 seq_len=116,
                 activation='Tanh',
                 device=torch.device('cpu'),
                 return_feat=False):
        super(ParallelModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.layers = layers
        self.observer = True
        self.activation = activation
        self.return_feat = return_feat
        self.acti = nn.Tanh()
        assert self.layers >= 2
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=self.in_dim,
                                        out_dim=self.out_dim,
                                        observer=True,
                                        activation=self.activation,
                                        device=device,
                                        return_feat=return_feat)
        if self.layers >= 2:
            self.state_layer2 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device,
                                            return_feat=return_feat)
        if self.layers >= 3:
            self.state_layer3 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device,
                                            return_feat=return_feat)
        if self.layers >= 4:
            self.state_layer4 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device,
                                            return_feat=return_feat)
        if self.layers >= 5:
            self.state_layer5 = StateNeuron(self.order,
                                            in_dim=self.in_dim,
                                            out_dim=self.out_dim,
                                            observer=True,
                                            activation=self.activation,
                                            device=device,
                                            return_feat=return_feat)
        self.fully_connected = nn.Linear(
            in_features=self.seq_len * self.layers, out_features=self.seq_len, bias=False).to(device)
        for weight in self.fully_connected.parameters():
            nn.init.uniform_(weight, 0, 0.5)

    def reset_parameter(self):
        """
        Reinitialize all trainable parameters

        :return: None
        """
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.5, 0.5)

    def count_parameters(self):
        """
        Count the total number of parameters

        :return: Total number of trainable parameters
        """
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None, feat_list=None):
        """
        Froward function for parallel DSSO-NN

        :param x: torch.tensor, input value
        :param y_obs: torch.tensor, observation value
        :param feat_list: list, for feature map generation if given
        :return: prediction made by the DSSO-NN
        """
        assert y_obs.shape[1] == self.seq_len
        if self.return_feat:
            assert type(feat_list) == list
        out1 = self.state_layer1(x, y_obs=y_obs, feat_list=feat_list)
        all = [out1]
        if self.layers >= 2:
            out2 = self.state_layer2(x, y_obs=y_obs, feat_list=feat_list)
            all.append(out2)
        if self.layers >= 3:
            out3 = self.state_layer2(x, y_obs=y_obs, feat_list=feat_list)
            all.append(out3)
        if self.layers >= 4:
            out4 = self.state_layer2(x, y_obs=y_obs, feat_list=feat_list)
            all.append(out4)
        if self.layers >= 5:
            out5 = self.state_layer2(x, y_obs=y_obs, feat_list=feat_list)
            all.append(out5)
        combined = torch.cat(all, 1)
        out = self.fully_connected(combined)

        return self.acti(out)


class StateModel(nn.Module):
    """
    Definition of the Serial DSSO-NN
    """

    def __init__(self, order, in_dim=1, out_dim=1,
                 layers=1, observer=False,
                 activation='Tanh',
                 device=torch.device('cpu'),
                 return_feat=False):
        super(StateModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.observer = observer
        self.activation = activation
        self.return_feat = return_feat
        if layers >= 2:
            self.state_space1 = StateSpace(2, in_dim=self.in_dim, activation='None', device=device)
        if layers >= 3:
            self.state_space2 = StateSpace(4, in_dim=2, activation='None', device=device)
        if layers >= 4:
            self.state_space3 = StateSpace(6, in_dim=4, activation='None', device=device)
        if layers >= 5:
            self.state_space4 = StateSpace(8, in_dim=6, activation='None', device=device)
        if layers >= 6:
            self.state_space5 = StateSpace(10, in_dim=8, activation='None', device=device)
        if layers >= 7:
            self.state_space6 = StateSpace(12, in_dim=10, activation='None', device=device)
        last_in_dim = in_dim if layers == 1 else 2 * layers - 2
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=last_in_dim,
                                        out_dim=self.out_dim,
                                        observer=True,
                                        activation=self.activation,
                                        device=device, return_feat=return_feat)

    def reset_parameter(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.5, 0.5)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None, feat_list=None):
        """
        Froward function for serial DSSO-NN

        :param x: torch.tensor, input value
        :param y_obs: torch.tensor, observation value
        :param feat_list: list, for feature map generation if given
        :return: prediction made by the DSSO-NN
        """
        if y_obs is not None:
            assert self.observer
        if self.return_feat:
            assert type(feat_list) == list
        if self.layers >= 2:
            out = self.state_space1(x)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        if self.layers >= 3:
            out = self.state_space2(out)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        if self.layers >= 4:
            out = self.state_space3(out)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        if self.layers >= 5:
            out = self.state_space4(out)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        if self.layers >= 6:
            out = self.state_space5(out)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        if self.layers >= 7:
            out = self.state_space6(out)
            if self.return_feat:
                feat_list.append(copy.deepcopy(out.detach()))
        out = self.state_layer1(out if self.layers > 1 else x, y_obs=y_obs, feat_list=feat_list)
        return out


# note for model name format:
# racefd_order_layer_type
# type: obs, nlobs, none

def main():
    """
    Main function including training and save the result

    :return: None
    """
    EPOCH = 20
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
                              seq_len=116, activation="None", device=device, return_feat=False)
    else:
        print('Unrecognized model structure:', structure)
        sys.exit(-2)

    # for name, param in model.named_parameters():
    #     print(name, param.size())

    print('Number of parameters: ', model.count_parameters())
    criterion = RMSELoss()

    data_gen = MyData()
    if data == "femto":
        train_set, test_set, show_set = data_gen.get_femto_data()
    else:
        train_set, test_set, show_set = data_gen.get_case_data(data)

    train_set, val_set = torch.utils.data.random_split(train_set, [600 - 150, 150])

    # training
    start = datetime.datetime.now()
    train_history = train(model,
                          criterion,
                          r2_loss,
                          epoch=EPOCH,
                          train_set=train_set,
                          val_set=val_set,
                          batch_size=BATCH_SIZE,
                          optimizer='Adam',
                          learning_rate=0.0005,
                          grad_clip=30)
    end = datetime.datetime.now()
    print('Time for training: ', (end - start).seconds, 'seconds')

    model_name = '../models/' + name + '.pt'
    torch.save(model, model_name)
    his_name = '../history/' + name + '.pt'
    his = torch.tensor([train_history.train_loss,
                        train_history.train_r2,
                        train_history.val_loss,
                        train_history.val_r2])
    torch.save(his, his_name)
    print('Name of the model: ', name)
    is_good = input("Continue validation? 1 to continue, 2 abort: ")
    is_good = int(is_good)
    if is_good == 2:
        sys.exit(0)
    # plot training curve
    train_fig = plt.figure(0)
    real_epoch = len(train_history.train_loss)
    epoch_vec = [i + 1 for i in range(real_epoch)]
    # train_fig.set_size_inches(12/2.54, 8/2.54)
    plt.plot(epoch_vec, train_history.train_loss)
    plt.plot(epoch_vec, train_history.train_r2)
    lgd = ['Training loss', 'Training R2-value']
    if note == "pltv":
        plt.plot(epoch_vec, train_history.val_loss)
        plt.plot(epoch_vec, train_history.val_r2)
        lgd.append("Validation loss")
        lgd.append("Validation R2-value")
    plt.legend(lgd, loc="upper right", fontsize=15)
    plt.ylim(0, 1.5)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    # plt.title('Training Process', fontsize=15)
    plt.grid(True)
    # if os.path.exists(fig_path + name + '_loss.png'):
    #     print('found old image, removing')
    #     os.remove(fig_path + name + '_loss.png')
    train_fig.savefig(fig_path + name + '_loss', dpi=300, bbox_inches='tight')

    # evaluation
    val_loss_obs, val_r2_obs, val_loss_wo, val_r2_wo = validation(
        model, test_set, criterion, origin=False, obs=True)
    print('validation loss with obs = ', val_loss_obs,
          '\nR2 loss with obs = ', val_r2_obs)
    print('validation loss wo obs = ', val_loss_wo,
          '\nR2 loss wo obs = ', val_r2_wo)
    # plot show dataset
    if data == "femto":
        dump3, dump4, dump5, dump4 = validation(
            model, show_set, criterion, num_data=3, origin=False, obs=True, show=True, fig_num=1)
    else:
        dump3, dump4, dump5, dump4 = validation(
            model, show_set, criterion, num_data=4, origin=False, obs=True, show=True, fig_num=1)

    print(config)
    # if os.path.exists(fig_path + name + '_val.png'):
    #     print('found old image, removing')
    #     os.remove(fig_path + name + '_val.png')
    plt.savefig(fig_path + name + '_val', dpi=300, bbox_inches='tight')
    plt.show()

    return 0


def recurrent_run():
    EPOCH = 20
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
    criterion = RMSELoss()

    data_gen = MyData()
    if data == "femto":
        train_set, test_set, show_set = data_gen.get_femto_data()
    else:
        train_set, test_set, show_set = data_gen.get_case_data(data)

    train_set, val_set = torch.utils.data.random_split(train_set, [600 - 150, 150])
    train_ls = []
    train_r2 = []
    val_ls = []
    val_r2 = []
    test_ls = []
    test_r2 = []

    for i in range(3):
        print("Training number ", i + 1)
        if structure == 'serial':
            model = StateModel(order, in_dim=2, out_dim=1, layers=layer,
                               observer=True, activation='None', device=device)
        elif structure == 'parallel':
            model = ParallelModel(order, in_dim=2, out_dim=1, layers=layer,
                                  seq_len=112, activation="None", device=device)
        else:
            print('Unrecognized model structure:', structure)
            sys.exit(-2)
        train_history = train(model,
                              criterion,
                              r2_loss,
                              epoch=EPOCH,
                              train_set=train_set,
                              val_set=val_set,
                              batch_size=BATCH_SIZE,
                              optimizer='Adam',
                              learning_rate=0.0005,
                              grad_clip=30,
                              print_loss=True)
        val_loss_obs, val_r2_obs, val_loss_wo, val_r2_wo = validation(
            model, test_set, criterion, origin=False, obs=True)
        # print('validation loss with obs = ', val_loss_obs,
        #         '\nR2 loss with obs = ', val_r2_obs)
        # print('validation loss wo obs = ', val_loss_wo,
        #         '\nR2 loss wo obs = ', val_r2_wo)
        train_ls.append(format(train_history.train_loss[-1], '.4f'))
        train_r2.append(format(train_history.train_r2[-1], '.4f'))
        val_ls.append(format(train_history.val_loss[-1], '.4f'))
        val_r2.append(format(train_history.val_r2[-1], '.4f'))
        test_ls.append(format(val_loss_obs, '.4f'))
        test_r2.append(format(val_r2_obs, '.4f'))
    print('Train loss all: ', train_ls, '\nTrain R2 all: ', train_r2,
          '\nVal loss all: ', val_ls, '\nVal R2 all: ', val_r2,
          '\nTest loss all: ', test_ls, '\nTest R2 all: ', test_r2)
    print(name)
    f = open('../results/' + name + '.txt', 'w')
    f.writelines(name)
    f.writelines('\n' + str(train_ls))
    f.writelines('\n' + str(train_r2))
    f.writelines('\n' + str(val_ls))
    f.writelines('\n' + str(val_r2))
    f.writelines('\n' + str(test_ls))
    f.writelines('\n' + str(test_r2))


if __name__ == '__main__':
    main()
    # recurrent_run()
