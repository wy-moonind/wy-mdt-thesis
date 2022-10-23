import torch
from torch import nn
from neurons import StateNeuron



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

        # out2 = self.state_layer2(out1)
        # out3 = self.state_layer3(out2, y_obs=y_obs)
        # out4 = self.state_layer4(out3)
        # out5 = self.state_layer5(out4, y_obs=y_obs)
