from validation import validation
import torch
from torch import nn
import sys


class TrainingHistory:

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_r2 = []


def train(model: nn.Module,
          criterion: nn.Module,
          metric,
          epoch: int,
          train_set: torch.utils.data.TensorDataset,
          val_set=None,
          batch_size=1,
          optimizer='Adam',
          learning_rate=1e-3,
          grad_clip=30):

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
    patience = 3
    trigger_times = 0
    for eph in range(epoch):
        loss_per_step = []
        r2_per_step = []
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
        if train_r2 < last_r2:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return history
        else:
            trigger_times = 0
        last_r2 = train_r2

    return history
