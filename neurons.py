import numpy as np
import torch
from torch import nn

activ_dict = {'ReLU': nn.ReLU(),
              'Tanh': nn.Tanh(),
              'Sigmoid': nn.Sigmoid(),
              'Softmax': nn.Softmax(dim=1),
              'Softplus': nn.Softplus()}


class StateNeuron(nn.Module):
    def __init__(self, order, in_dim=1, out_dim=1, activation='Tanh', device=torch.device('cpu')):
        super(StateNeuron, self).__init__()
        self.seq_len = 0
        self.order = order
        self.inp_dim = in_dim
        self.out_dim = out_dim
        self.weight_a = nn.Parameter(torch.ones((self.order, self.order), device=device))
        self.weight_b = nn.Parameter(torch.ones((self.order, self.inp_dim), device=device))
        self.weight_c = nn.Parameter(torch.ones((self.out_dim, self.order), device=device))
        self.weight_d = nn.Parameter(torch.ones((self.out_dim, self.inp_dim), device=device))
        self.weight_l = nn.Parameter(torch.ones((self.order, 1), device=device))
        self.activation = activ_dict.get(activation)
        self.transfer_gpu = False
        self.device = device
        if device != torch.device('cpu'):
            self.transfer_gpu = True
        self.reset_parameter()

    def reset_parameter(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.7, 0.7)

    def forward(self, u, y_obs=None):
        if u.ndim > 2:
            u = torch.squeeze(u)
        self.seq_len = u.shape[1]
        y_hat = torch.FloatTensor(self.out_dim, 1)
        if y_obs is not None:
            assert y_obs.shape[1] == u.shape[1]
            y_obs = y_obs.float()
            y_obs = y_obs.view(1, self.seq_len)
        u = u.float()
        u = u.view(self.inp_dim, self.seq_len)
        hidden = torch.FloatTensor(self.order, 1)
        nn.init.constant_(hidden, 1)
        noise = torch.FloatTensor(1)
        if self.transfer_gpu:
            y_hat = y_hat.cuda(self.device)
            hidden = hidden.cuda(self.device)
            noise = noise.cuda(self.device)
        for i in range(self.seq_len):
            nn.init.uniform_(noise, 0, 0.1)
            if self.inp_dim > 1:
                y_t = torch.mm(self.weight_c, hidden) + torch.mm(self.weight_d, u[:, [i]]) + noise
                x_t1 = torch.mm(self.weight_a, hidden) + torch.mm(self.weight_b, u[:, [i]])
            else:
                y_t = torch.mm(self.weight_c, hidden) + self.weight_d * u[:, i] + noise
                # y_t = torch.mm(self.normal_c, hidden)
                x_t1 = torch.mm(self.weight_a, hidden) + self.weight_b * u[:, i]
            if y_obs is not None:
                x_t1 += self.weight_l * (y_obs[0, i] - y_t)
            hidden = x_t1
            if i == 0:
                y_hat = y_t
            else:
                y_hat = torch.cat((y_hat, y_t), 1)
        return self.activation(y_hat) * 3

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

