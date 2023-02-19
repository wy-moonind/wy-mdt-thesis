import torch
from torch import nn
import matplotlib.pyplot as plt
# extra import
from loss_func import RMSELoss, r2_loss


def validation(model: nn.Module,
               val_set: torch.utils.data.TensorDataset,
               criterion: nn.Module,
               num_data=5,
               origin=False,  # print without observer result
               obs=False,  # print observer result
               show=False,
               fig_num=1):
    """
    Implementation of the validation process (for validation and test)
    :param model: torch.nn.Module, model to be validated
    :param val_set: torch.utils.data.TensorDataset, validation set
    :param criterion: loss function
    :param num_data: number of data to be shown
    :param origin: bool, with or without observer?
    :param obs: bool, with or without observer?
    :param show: bool, show validation figures?
    :param fig_num:
    :return: float, the averaged metrics
    """
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)
    if show:
        fig = plt.figure(fig_num)
        fig.set_size_inches(22/2.54, 16/2.54, forward=True)
        # fig.suptitle('Randomly selected validation data')
    if origin:
        val_loss_wo = []
        val_r2_wo = []
    val_loss_obs = []
    val_r2_obs = []
    for idx, (batch_x, batch_obs, batch_y) in enumerate(val_loader):
        if obs:
            pred_obs = model(batch_x, y_obs=batch_obs)
            pred_obs = pred_obs.cpu()
            temp_val_r2 = r2_loss(pred_obs.cuda(), batch_y)
            temp_loss_obs = criterion(pred_obs.cuda(), batch_y).item()
            val_loss_obs.append(temp_loss_obs)
            val_r2_obs.append(temp_val_r2)
        if origin:
            pred = model(batch_x).cpu()
            temp_loss = criterion(pred.cuda(), batch_y).item()
            temp_r2 = r2_loss(pred.cuda(), batch_y)
            val_loss_wo.append(temp_loss)
            val_r2_wo.append(temp_r2)

        if show:
            assert idx <= num_data
            plt.subplot(num_data, 1, idx + 1)
            plt.plot(batch_y.detach().cpu().numpy().T)
            lgd_vec = ["Target"]
            if obs:
                plt.plot(pred_obs.detach().numpy().T)
                lgd_vec.append("Prediction")
            if origin:
                plt.plot(pred.detach().numpy().T)
                lgd_vec.append("Prediction")
            plt.legend(lgd_vec, loc="upper right")
            plt.ylabel('Acceleration', fontsize=8)
            plt.xlabel('Timesteps', fontsize=10)
            plt.grid(True)
    avg_val_loss_wo = None
    avg_val_r2_wo = None
    avg_val_loss_obs = None
    avg_val_r2_obs = None
    if origin:
        avg_val_loss_wo = float(sum(val_loss_wo) / len(val_loss_wo))
        avg_val_r2_wo = float(sum(val_r2_wo) / len(val_r2_wo))
    if obs:
        avg_val_loss_obs = float(sum(val_loss_obs) / len(val_loss_obs))
        avg_val_r2_obs = float(sum(val_r2_obs) / len(val_r2_obs))

    return avg_val_loss_obs, avg_val_r2_obs, avg_val_loss_wo, avg_val_r2_wo

