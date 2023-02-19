from validation import validation
import torch
from torch import nn
import sys


class TrainingHistory:
    """
    Definition of the training history
    """

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_r2 = []
        self.val_r2 = []


def train(model: nn.Module,
          criterion: nn.Module,
          metric,
          epoch: int,
          train_set: torch.utils.data.TensorDataset,
          val_set=None,
          batch_size=1,
          optimizer='Adam',
          learning_rate=1e-3,
          grad_clip=30,
          print_loss=True):
    """
    Implementation of the training process
    :param model: torch.nn.Module, the model to be trained
    :param criterion: loss function
    :param metric: function of the R2 score
    :param epoch: Maximum number of epochs
    :param train_set: torch.utils.data.TensorDataset, training set
    :param val_set: torch.utils.data.TensorDataset, validation set
    :param batch_size: default to be 1
    :param optimizer: string, name of the optimizer
    :param learning_rate:
    :param grad_clip: 30
    :param print_loss: whether print the loss during the training
    :return: Training history of each step
    """

    history = TrainingHistory()
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    if optimizer == 'Adam':
        opti = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=0.1)
    else:
        print('Unexpected Optimizer Type!')
        sys.exit(-1)
    last_loss = 1e20
    last_r2 = -1e20
    patience = 1
    trigger_times = 0
    restart = True
    while restart:
        restart = False
        for eph in range(epoch):
            loss_per_step = []
            r2_per_step = []
            restart_inside = False
            for step, (batch_x, batch_yobs, batch_y) in enumerate(train_loader):
                x = torch.autograd.Variable(batch_x)
                y_init = torch.autograd.Variable(batch_yobs)
                y = torch.autograd.Variable(batch_y)
                if model.observer:
                    output = model(x, y_obs=y_init)
                else:
                    output = model(x)
                loss = criterion(output, y)
                temp_r2 = metric(output, y)

                loss_per_step.append(loss.item())
                if torch.isnan(loss):
                    print('NaN value, reinitialize parameters')
                    restart_inside = True
                    break
                r2_per_step.append(temp_r2)
                # print(step, ":", loss.item(), temp_r2)
                opti.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                opti.step()
            if restart_inside:
                restart = True
                model.reset_parameter()
                break
            train_loss = float(sum(loss_per_step) / len(loss_per_step))
            train_r2 = float(sum(r2_per_step) / len(r2_per_step))
            history.train_loss.append(train_loss)
            history.train_r2.append(train_r2)
            if print_loss:
                print('Epoch ', eph + 1,
                      '\nTraining loss = ', train_loss, '; train r2 = ', train_r2)

            # validation
            if val_set is not None:
                val_loss, val_r2, dump1, dump2 = validation(model, val_set, criterion=criterion, origin=False, obs=True)
                # history.val_loss.append(val_loss)
                # history.val_r2.append(val_r2)
                history.val_loss.append(val_loss)
                history.val_r2.append(val_r2)
                if print_loss:
                    print('Validation loss = ', val_loss, '; val r2 = ', val_r2)

            # Early-stopping
            if eph >= 2:
                if train_r2 < last_r2:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print('Early stopping!\nEpoch:', eph + 1)
                        return history
                else:
                    trigger_times = 0
                last_r2 = train_r2

    return history
