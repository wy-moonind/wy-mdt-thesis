from model import StateModel, TrainingHistory
from loss_func import RMSELoss, r2_loss
from validation import validation
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from data import MyData
import sys


def train(model: nn.Module,
          criterion: nn.Module,
          epoch: int,
          train_set: torch.utils.data.TensorDataset,
          val_set=None,
          batch_size=1,
          optimizer='Adam',
          learning_rate=0.001,
          grad_clip=30):

    history = TrainingHistory()
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    if optimizer == 'Adam':
        opti = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    else:
        print('Unexpected Optimizer Type!')
        sys.exit(-1)
    last_loss = 1e20
    patience = 3
    trigger_times = 0
    for eph in range(epoch):
        loss_per_step = []
        r2_per_step = []
        for step, (batch_x, batch_yobs, batch_y) in enumerate(train_loader):
            x = torch.autograd.Variable(batch_x)
            y_init = torch.autograd.Variable(batch_yobs)
            y = torch.autograd.Variable(batch_y)

            output = model(x, y_obs=y_init)
            # output = model(x)
            loss = criterion(output, y)
            temp_r2 = r2_loss(output, y)

            loss_per_step.append(loss.item())
            r2_per_step.append(temp_r2)
            opti.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opti.step()

        train_loss = sum(loss_per_step) / len(loss_per_step)
        train_r2 = sum(r2_per_step) / len(r2_per_step)
        history.train_loss.append(train_loss)
        history.train_r2.append(train_r2)
        print('Epoch ', eph + 1,
              ' Training loss = ', train_loss)

        # validation
        if val_set is not None:
            val_loss, val_r2 = validation(model, val_set, criterion=criterion)
            history.val_loss.append(val_loss)
            print('Validation loss = ', val_loss, 'R2 loss = ', val_r2)

        # Early-stopping
        if train_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return history
        else:
            trigger_times = 0
        last_loss = train_loss

    return history


def main():
    EPOCH = 250
    BATCH_SIZE = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # start with main function
    model = StateModel(5, in_dim=2, out_dim=1, observer=True, device=device)
    print('Number of parameters: ', model.count_parameters())
    # model = model.cuda(device)
    criterion = RMSELoss()

    data_gen = MyData()
    # dataset = data_gen.get_outer_data()
    dataset = data_gen.get_case07()

    train_set, val_set = torch.utils.data.random_split(dataset, [25, 4])
    # training
    train_history = train(model,
                          criterion,
                          epoch=EPOCH,
                          train_set=train_set,
                          batch_size=BATCH_SIZE,
                          optimizer='Adam',
                          learning_rate=0.001,
                          grad_clip=30)
    name = 'test00'
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
    plt.savefig('../figs/test/' + name + '_loss')

    # evaluation
    val_loss, val_r2 = validation(model, val_set, criterion, num_data=4, origin=True, show=True, fig_num=1)
    print('validation loss = ', val_loss, '\nR2 loss = ', val_r2)
    plt.savefig('../figs/test/' + name + '_val')

    plt.show()

    return 0


if __name__ == '__main__':
    main()
