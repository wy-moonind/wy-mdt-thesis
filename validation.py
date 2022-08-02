import torch
from torch import nn
import matplotlib.pyplot as plt
from loss_func import r2_loss
# extra import
from data import MyData
from loss_func import RMSELoss, r2_loss


def validation(model: nn.Module,
               val_set: torch.utils.data.TensorDataset,
               criterion: nn.Module,
               num_data=5,
               origin=False,  # print without observer result
               obs=False,  # print observer result
               show=False,
               fig_num=1):
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=1, shuffle=False)
    if show:
        fig = plt.figure(fig_num)
        fig.set_size_inches(12, 7, forward=True)
        fig.suptitle('Randomly selected validation data')
    if origin:
        val_loss_wo = []
        val_r2_wo = []
    val_loss_obs = []
    val_r2_obs = []
    for idx, (batch_x, batch_obs, batch_y) in enumerate(val_loader):
        if obs:
            pred_obs = model(batch_x, y_obs=batch_obs).cpu()
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
            lgd_vec = ["target"]
            if obs:
                plt.plot(pred_obs.detach().numpy().T)
                lgd_vec.append("prediction with obs")
            if origin:
                plt.plot(pred.detach().numpy().T)
                lgd_vec.append("prediction")
            plt.legend(lgd_vec)
            plt.ylabel('Acceleration')
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


if __name__ == '__main__':
    model = torch.load('../models/case_fd21/case_layer3_order13.pt')
    data_gen = MyData()
    # dataset = data_gen.get_outer_data()
    dataset = data_gen.get_case_data()

    train_set, val_set = torch.utils.data.random_split(dataset, [41, 4])

    criterion = RMSELoss()

    val_loss, val_r2 = validation(
        model, val_set, criterion, num_data=4, show=True, fig_num=0)
    plt.xlabel('Timestep')

    plt.show()
