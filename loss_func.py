import torch
from torch import nn
from loss import soft_dtw
from loss import path_soft_dtw


class AvgLoss(nn.Module):

    def __init__(self):
        super(AvgLoss, self).__init__()

    def forward(self, x, y):
        # if (x.shape[0] == y.shape[1] and x.shape[1] == y.shape[0]):
        #     x = torch.transpose(x, 0, 1)
        assert x.shape == y.shape
        # loss from data to data, euclidean distance
        # l_dist = torch.linalg.norm(x - y)
        # loss from frequency domain
        # return l_dist.detach()
        return torch.sqrt(torch.linalg.norm(x - y))


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return float(r2)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))


class DilateLoss(nn.Module):
    def __init__(self, alpha, gamma, device):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, outputs, targets):
        # outputs, targets: shape (batch_size, N_output, 1)
        batch_size, N_output = outputs.shape[0:2]
        loss_shape = 0
        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        D = torch.zeros((batch_size, N_output, N_output)).to(self.device)
        for k in range(batch_size):
            Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = softdtw_batch(D, self.gamma)

        path_dtw = path_soft_dtw.PathDTWBatch.apply
        path = path_dtw(D, self.gamma)
        Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(self.device)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss, loss_shape, loss_temporal
