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
               origin=False,
               show=False,
               fig_num=1):
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    if show:
        fig = plt.figure(fig_num)
        fig.set_size_inches(16, 9, forward=True)
        fig.suptitle('Randomly selected validation data')
    if origin:
        val_loss = []
        val_r2 = []
    val_loss_obs = []
    val_r2_per_step = []
    for idx, (batch_x, batch_obs, batch_y) in enumerate(val_loader):
        pred_obs = model(batch_x, y_obs=batch_obs).cpu()
        pred = model(batch_x).cpu()
        if origin:
            temp_loss = criterion(pred.cuda(), batch_y).item()
            temp_r2 = r2_loss(pred.cuda(), batch_y)
            val_loss.append(temp_loss)
            val_r2.append(temp_r2)

        temp_val_r2 = r2_loss(pred_obs.cuda(), batch_y)
        temp_loss_obs = criterion(pred_obs.cuda(), batch_y).item()

        val_loss_obs.append(temp_loss_obs)
        val_r2_per_step.append(temp_val_r2)

        if show:
            assert idx <= num_data
            plt.subplot(num_data, 1, idx + 1)
            plt.plot(batch_y.detach().cpu().numpy().T)
            plt.plot(pred_obs.detach().numpy().T)
            if origin:
                plt.plot(pred.detach().numpy().T)
                plt.legend(["raw", "prediction with obs", "prediction"]) #
            else:
                plt.legend(["raw", "prediction"])
            plt.ylabel('Acceleration')
            plt.grid(True)
    if origin:
        avg_val_loss = sum(val_loss) / len(val_loss)
        print('validation loss without obs: ', avg_val_loss)
        avg_val_r2_wo = sum(val_r2) / len(val_r2)
        print('validation r2 without obs: ', float(avg_val_r2_wo))
    avg_val_loss_obs = sum(val_loss_obs) / len(val_loss_obs)
    avg_val_r2 = sum(val_r2_per_step) / len(val_r2_per_step)

    return avg_val_loss_obs, float(avg_val_r2)


if __name__ == '__main__':
    model = torch.load('./models/case_layer5_order13.pt')
    data_gen = MyData()
    # dataset = data_gen.get_outer_data()
    dataset = data_gen.get_case_data()

    train_set, val_set = torch.utils.data.random_split(dataset, [41, 4])

    criterion = RMSELoss()

    val_loss, val_r2 = validation(model, val_set, criterion, num_data=4, show=True, fig_num=0)
    plt.xlabel('Timestep')

    plt.show()
