from neurons import StateNeuron
from validation import validation
from train import train
from loss_func import RMSELoss, r2_loss
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from data import MyData
import sys

class StateModel(nn.Module):

    def __init__(self, order, in_dim=1, out_dim=1, observer=False, activation='Tanh', device=torch.device('cpu')):
        super(StateModel, self).__init__()
        self.order = order
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.observer = observer
        self.activation = activation
        self.state_layer1 = StateNeuron(self.order,
                                        in_dim=self.in_dim,
                                        out_dim=self.out_dim,
                                        activation=self.activation,
                                        device=device)
        # self.state_layer2 = StateNeuron(self.order,
        #                                 in_dim=2,
        #                                 out_dim=self.out_dim,
        #                                 device=device)
        # self.state_layer3 = StateNeuron(self.order,
        #                                 in_dim=4,
        #                                 out_dim=self.out_dim,
        #                                 device=device)
        # self.state_layer4 = StateNeuron(self.order,
        #                                 in_dim=6,
        #                                 out_dim=self.out_dim,
        #                                 device=device)
        # self.state_layer5 = StateNeuron(self.order,
        #                                 in_dim=8,
        #                                 out_dim=self.out_dim,
        #                                 device=device)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None):
        if y_obs is not None:
            assert self.observer
        out1 = self.state_layer1(x, y_obs=y_obs)
        # out2 = self.state_layer2(out1)
        # out3 = self.state_layer3(out2)
        # out4 = self.state_layer4(out3)
        # out5 = self.state_layer5(out4)
        return out1

def main():
    EPOCH = 250
    BATCH_SIZE = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert len(sys.argv) == 4
    order = int(sys.argv[1])
    name = sys.argv[2]  # 'test_07_tanh_7_4'
    data = sys.argv[3] # 'fd07_outer'

    # start with main function
    model = StateModel(order, in_dim=2, out_dim=1, observer=False, activation='None', device=device)
    print('Number of parameters: ', model.count_parameters())
    # model = model.cuda(device)
    criterion = RMSELoss()

    data_gen = MyData()
    train_set, test_set, train_size = data_gen.get_case_data(data)
    print("Dataset size: ", train_size)

    # train_set, val_set = torch.utils.data.random_split(train_set, [41, 4])
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

    model_name = '../models/test/' + name + '.pt'
    torch.save(model, model_name)
    # plot training curve
    plt.figure(0)
    plt.plot(train_history.train_loss)
    plt.plot(train_history.train_r2)
    plt.ylim(0, 1)
    plt.legend(['Training loss', 'Training R2-value'])
    plt.xlabel('Epoch')
    plt.title('Training Process')
    plt.grid(True)
    plt.savefig('../figs/test/' + name + '_loss', dpi=300)

    # evaluation
    val_loss_obs, val_r2_obs, val_loss_wo, val_r2_wo = validation(model, test_set, criterion, num_data=4, origin=True, obs=False, show=True, fig_num=1)
    print('validation loss with obs = ', val_loss_obs, '\nR2 loss with obs = ', val_r2_obs)
    print('validation loss wo obs = ', val_loss_wo, '\nR2 loss wo obs = ', val_r2_wo)
    plt.savefig('../figs/test/' + name + '_val', dpi=300)

    plt.show()

    return 0


if __name__ == '__main__':
    main()
