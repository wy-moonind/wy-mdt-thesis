class ParallelModel(nn.Module):

    def __init__(self, order, layers=2, 
                in_dim=1, out_dim=1, 
                seq_len=112, 
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
            in_features=self.seq_len*self.layers, out_features=self.seq_len, bias=False).to(device)
        for weight in self.fully_connected.parameters():
            nn.init.uniform_(weight, 0, 0.5)

    def reset_parameter(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.5, 0.5)

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x, y_obs=None):
        assert y_obs.shape[1] == self.seq_len
        feat_list = []
        out1, feat1 = self.state_layer1(x, y_obs=y_obs)
        all = [out1]
        feat_list.append(feat1.detach())
        if self.layers >= 2: 
            out2, feat2 = self.state_layer2(x, y_obs=y_obs)
            all.append(out2)
            feat_list.append(feat2.detach())
        if self.layers >= 3: 
            out3, feat3 = self.state_layer2(x, y_obs=y_obs)
            all.append(out3)
            feat_list.append(feat3.detach())
        if self.layers >= 4: 
            out4, feat4 = self.state_layer2(x, y_obs=y_obs)
            all.append(out4)
            feat_list.append(feat4.detach())
        if self.layers >= 5: 
            out5, feat5 = self.state_layer2(x, y_obs=y_obs)
            all.append(out5)
            feat_list.append(feat5.detach())
        combined = torch.cat(all, 1)
        out = self.fully_connected(combined)

        return self.acti(out), feat_list


class StateModel(nn.Module):

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
        if layers>=7:
            self.state_space6 = StateSpace(12, in_dim=10, activation='None', device=device)
        last_in_dim = in_dim if layers==1 else 2*layers-2
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

    def forward(self, x, y_obs=None):
        feat_list = []
        if y_obs is not None:
            assert self.observer
        if self.layers >=2:
            out = self.state_space1(x)
            temp1 = out.clone().detach()
            feat_list.append(temp1)
        if self.layers >=3:
            out = self.state_space2(out)
            temp2 = out.clone().detach()
            feat_list.append(temp2)
        if self.layers >=4:
            out = self.state_space3(out)
            temp3 = out.clone().detach()
            feat_list.append(temp3)
        if self.layers >=5:
            out = self.state_space4(out)
            temp4 = out.clone().detach()
            feat_list.append(temp4)
        if self.layers >=6:
            out = self.state_space5(out)
        if self.layers>=7:
            out = self.state_space6(out)
        if self.return_feat:
            out, last_feat = self.state_layer1(out if self.layers>1 else x, y_obs=y_obs)
            feat_list.append(last_feat)
            return out, feat_list
        else:
            out = self.state_layer1(out if self.layers>1 else x, y_obs=y_obs)
            return out

class StateNeuron(nn.Module):
    def __init__(self, order, in_dim=1, out_dim=1,
                 observer=False, activation='Tanh',
                 device=torch.device('cpu'),
                 return_feat=False):
        super(StateNeuron, self).__init__()
        self.seq_len = 0
        self.order = order
        self.inp_dim = in_dim
        self.out_dim = out_dim
        self.observer = observer
        self.return_feat = return_feat
        self.weight_a = nn.Parameter(torch.ones(
            (self.order, self.order), device=device))
        self.weight_b = nn.Parameter(torch.ones(
            (self.order, self.inp_dim), device=device))
        self.weight_c = nn.Parameter(torch.ones(
            (self.out_dim, self.order), device=device))
        self.weight_d = nn.Parameter(torch.ones(
            (self.out_dim, self.inp_dim), device=device))
        if self.observer:
            self.weight_l = nn.Parameter(
                torch.ones((self.order, 1), device=device))
            self.weight_alpha = nn.Parameter(
                torch.ones((self.order, 1), device=device))
            # self.fixed_delta = nn.Parameter(torch.tensor(0.5, device=device)) # trainable delta
            self.fixed_delta = torch.tensor(0.5, device=device) # fixed delta
        self.activation = activ_dict.get(activation)
        self.transfer_gpu = False
        self.device = device
        if device != torch.device('cpu'):
            self.transfer_gpu = True
        self.reset_parameter()

    def reset_parameter(self):
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.7, 0.7)

    def nonlinear(self, error):
        # fal function
        assert self.observer
        if(torch.abs(error) <= self.fixed_delta):
            return torch.div(error, torch.pow(input=self.fixed_delta, exponent=(1-self.weight_alpha)))
        else:
            return torch.pow(input=torch.abs(error), exponent=self.weight_alpha) * torch.sign(error)

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
                y_t = torch.mm(self.weight_c, hidden) + \
                    torch.mm(self.weight_d, u[:, [i]]) + noise
                x_t1 = torch.mm(self.weight_a, hidden) + \
                    torch.mm(self.weight_b, u[:, [i]])
            else:
                y_t = torch.mm(self.weight_c, hidden) + \
                    self.weight_d * u[:, i] + noise
                # y_t = torch.mm(self.normal_c, hidden)
                x_t1 = torch.mm(self.weight_a, hidden) + \
                    self.weight_b * u[:, i]
            if y_obs is not None:
                # x_t1 += self.weight_l * (y_obs[0, i] - y_t)
                x_t1 += self.weight_l * self.nonlinear(y_obs[0, i] - y_t)
            if i == 0:
                y_hat = y_t
                x_hat = hidden
            else:
                y_hat = torch.cat((y_hat, y_t), 1)
                x_hat = torch.cat((x_hat, hidden), 1)
            hidden = x_t1
        if self.return_feat:
            return self.activation(y_hat), x_hat
        return self.activation(y_hat)