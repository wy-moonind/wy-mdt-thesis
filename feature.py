from engine import StateModel, ParallelModel
from math import comb
from turtle import forward
from neurons import StateNeuron, StateSpace
from validation import validation
from train import train
from loss_func import RMSELoss, r2_loss
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from data import MyData
import sys
import torch.utils.data as Data

criterion = RMSELoss()

def step1():
    EPOCH = 20
    BATCH_SIZE = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    order = 15
    layer = 1

    # train and save model
    model = StateModel(order, in_dim=2, out_dim=1, layers=layer,
                           observer=True, activation='None', device=device, return_feat=True)
    # model = ParallelModel(order, in_dim=2, out_dim=1, layers=layer,
    #                           seq_len=116, activation="None", device=device, return_feat=True)

    criterion = RMSELoss()

    data_gen = MyData()
    train_set, test_set, show_set = data_gen.get_case_data("outer21")

    train_set, val_set = torch.utils.data.random_split(train_set, [600-150, 150])
    train_history = train(model,
                            criterion,
                            r2_loss,
                            epoch=EPOCH,
                            train_set=train_set,
                            # val_set=val_set,
                            batch_size=BATCH_SIZE,
                            optimizer='Adam',
                            learning_rate=0.0005,
                            grad_clip=30)
    torch.save(model, 'show_feat_outer21_1_15_lso.pt')

    
def step2():
    model = torch.load('../models/show_feat_outer21_1_15_nlso.pt')
    model1 = torch.load('../models/show_feat_outer21_1_15_lso.pt')
    data_gen = MyData()
    train_set, test_set, show_set = data_gen.get_case_data("outer21")
    show_featmap = []
    show_featmap1 = []
    show_loader = torch.utils.data.DataLoader(
        dataset=show_set, batch_size=1, shuffle=False)
    for idx, (batch_x, batch_obs, batch_y) in enumerate(show_loader):
        pred = model(batch_x, y_obs=batch_obs, feat_list=show_featmap)
        pred1 = model1(batch_x, y_obs=batch_obs, feat_list=show_featmap1)
        if idx > 0:
            break
    print(show_featmap[0].shape)

    fig = plt.figure(figsize=(8,4))

    
    ax = fig.add_subplot(2, 1, 1)
    ax.set_title('Non-linear State Observer')
    ax.imshow(show_featmap[0].cpu().detach())

    ax = fig.add_subplot(2, 1, 2)
    ax.set_title('Linear State Observer')
    ax.imshow(show_featmap1[0].cpu().detach())

    plt.xticks(size=10)
    plt.yticks(size=10)

    # plt.savefig('../figs/nlso_lso_feature', dpi=300, bbox_inches='tight')
    plt.show()

def step3():
    model = torch.load('../models/show_feat_outer21_5_15.pt')
    model1 = torch.load('../models/show_feat_outer21_5_15_par.pt')
    data_gen = MyData()
    train_set, test_set, show_set = data_gen.get_case_data("outer21")
    show_featmap = []
    show_featmap1 = []
    show_loader = torch.utils.data.DataLoader(
        dataset=show_set, batch_size=1, shuffle=False)
    for idx, (batch_x, batch_obs, batch_y) in enumerate(show_loader):
        pred = model(batch_x, y_obs=batch_obs, feat_list=show_featmap)
        pred1 = model1(batch_x, y_obs=batch_obs, feat_list=show_featmap1)
        if idx>0:
            break

    fig = plt.figure(figsize=(8,7))

    for i in range(5):
        ax = fig.add_subplot(5, 1, i+1)
        ax.set_title('Layer'+str(i+1))
        ax.imshow(show_featmap[i].cpu().detach())
    plt.xticks(size=10)
    plt.yticks(size=10)

    # plt.savefig('../figs/serial_feature', dpi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(8,8))
    for j in range(5):
        ax = fig.add_subplot(5, 1, j+1)
        ax.set_title('Layer'+str(j+1))
        ax.imshow(show_featmap1[j].cpu().detach())

    plt.xticks(size=10)
    plt.yticks(size=10)

    # plt.savefig('../figs/parallel_feature', dpi=300, bbox_inches='tight')
    plt.show()

def step4():
    model = torch.load('../models/outer21_15_1_serial_woobs.pt')
    # outer21 test
    test_u = torch.load('../data/case_data_copy/test/test_u_outer21.pt')
    test_y = torch.load('../data/case_data_copy/test/test_y_outer21.pt')
    test_set1 = Data.TensorDataset(test_u.cuda(),
                                      test_y.cuda())
    loader_outer21 = torch.utils.data.DataLoader(
        dataset=test_set1, batch_size=1, shuffle=False)
    out_list = []
    target_list = []
    for idx, (batch_x, batch_y) in enumerate(loader_outer21):
        pred = model(batch_x)
        out_list.append(pred.cpu().detach())
        target_list.append(batch_y.cpu())
    
    plt.figure(figsize=(8,4.5))
    plt.plot(out_list[0].numpy().T, color='blue')
    plt.plot(target_list[0].numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    plt.savefig('../figs/outer21_val_woobs', dpi=300, bbox_inches='tight')

    model1 = torch.load('../models/outer07_15_1_serial_woobs.pt')
    # outer21 test
    test_u = torch.load('../data/case_data_copy/test/test_u_outer07.pt')
    test_y = torch.load('../data/case_data_copy/test/test_y_outer07.pt')
    test_set1 = Data.TensorDataset(test_u.cuda(),
                                      test_y.cuda())
    loader_outer07 = torch.utils.data.DataLoader(
        dataset=test_set1, batch_size=1, shuffle=False)
    out_list1 = []
    target_list1 = []
    for idx, (batch_x, batch_y) in enumerate(loader_outer07):
        pred = model1(batch_x)
        out_list1.append(pred.cpu().detach())
        target_list1.append(batch_y.cpu())
    
    plt.figure(figsize=(8,4.5))
    plt.plot(out_list1[0].numpy().T, color='blue')
    plt.plot(target_list1[0].numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    plt.savefig('../figs/outer07_val_woobs', dpi=300, bbox_inches='tight')

    plt.show()

def step5():
    model = torch.load('../models/outer07_15_5_parallel_newdata.pt')
    # model = torch.load('../models/show_feat_outer21_5_15_par.pt')
    # model.seq_len = 112
    # outer21 test
    test_u = torch.load('../data/case_data/test/test_u_outer07.pt')
    test_y = torch.load('../data/case_data/test/test_y_outer07.pt')
    test_set1 = Data.TensorDataset(test_u.cuda(),
                                      test_y.cuda(),
                                      test_y.cuda())
    loader_outer21 = torch.utils.data.DataLoader(
        dataset=test_set1, batch_size=1, shuffle=False)
    out_list = []
    target_list = []
    for idx, (batch_x, batch_obs, batch_y) in enumerate(loader_outer21):
        pred, dump = model(batch_x, y_obs=batch_obs)
        out_list.append(pred.cpu().detach())
        target_list.append(batch_y.cpu())
        r2 = r2_loss(pred.cpu().detach(), batch_y.cpu())
        if idx == 0:
            print(r2)
            break
    
    plt.figure(figsize=(8,4.5))
    plt.plot(out_list[0].numpy().T, color='blue')
    plt.plot(target_list[0].numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    plt.savefig('../figs/outer07_val_par', dpi=300, bbox_inches='tight')

    # model1 = torch.load('../models/outer07_15_1_serial_linear.pt')
    # # outer21 test
    # test_u = torch.load('../data/case_data_copy/test/test_u_outer07.pt')
    # test_y = torch.load('../data/case_data_copy/test/test_y_outer07.pt')
    # test_set1 = Data.TensorDataset(test_u.cuda(),
    #                                     test_y.cuda(),
    #                                   test_y.cuda())
    # loader_outer07 = torch.utils.data.DataLoader(
    #     dataset=test_set1, batch_size=1, shuffle=False)
    # out_list1 = []
    # target_list1 = []
    # for idx, (batch_x,batch_obs, batch_y) in enumerate(loader_outer07):
    #     pred = model1(batch_x, y_obs=batch_obs)
    #     out_list1.append(pred.cpu().detach())
    #     target_list1.append(batch_y.cpu())
    
    # plt.figure(figsize=(8,4.5))
    # plt.plot(out_list1[0].numpy().T, color='blue')
    # plt.plot(target_list1[0].numpy().T, color='red')
    # plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    # plt.xticks(size=12)
    # plt.yticks(size=12)
    # plt.xlabel('Timesteps',fontsize=15)
    # plt.ylabel('Normalized Acceleration',fontsize=15)
    # plt.savefig('../figs/outer07_val_linear', dpi=300, bbox_inches='tight')

    plt.show()

def step6():
    
    name = 'outer21'
    history = torch.load('../history/'+name+'_15_5_serial_newdata.pt')
    model = torch.load('../models/'+name+'_15_5_serial_newdata.pt')
    test_u = torch.load('../data/case_data/test/show_u_'+name+'.pt')
    test_y = torch.load('../data/case_data/test/show_y_'+name+'.pt')
    # test_u = torch.load('../data/femto_data/test/show_u_all.pt')
    # test_y = torch.load('../data/femto_data/test/show_y_all.pt')

    real_epoch = len(history[0, :])
    fig = plt.figure(figsize=(6,4))
    epoch_vec = [i+1 for i in range(real_epoch)]
    ax1 = fig.add_subplot(111)
    ax1.plot(epoch_vec, history[0, :], label="Training loss", color="blue")
    ax1.plot(epoch_vec, history[2, :], label="Validation loss", color="green")
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("RMSE loss", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(axis="y")
    ax2 = ax1.twinx()
    ax2.plot(epoch_vec, history[1, :], label="Training R2-value", color="darkorange")
    ax2.plot(epoch_vec, history[3, :], label="Validation R2-value", color="red")
    ax2.set_ylabel("$R^2$ score", fontsize=12)
    ax2.legend(loc='upper right')
    ax1.set_ylim(0, 1.5)
    ax1.set_xlim(0, 12)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 12)
    fig.savefig('../figs/'+name+'_loss', dpi=300, bbox_inches='tight')
    
    testset = Data.TensorDataset(test_u.cuda(),
                                        test_y.cuda(),
                                      test_y.cuda())
    loader= torch.utils.data.DataLoader(
        dataset=testset, batch_size=1, shuffle=False)
    rus = []
    y = []
    for idx, (batch_x,batch_obs, batch_y) in enumerate(loader):
        pred = model(batch_x, y_obs=batch_obs)
        rus.append(pred)
        y.append(batch_y)
    plt.figure(figsize=(8,4.5))
    plt.plot(rus[0].detach().cpu().numpy().T, color='blue')
    plt.plot(y[0].detach().cpu().numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    plt.savefig('../figs/'+name+'_val', dpi=300, bbox_inches='tight')

    plt.show()

def step7():
    test_u = torch.load('../data/case_data_copy/test/test_u_outer21.pt')
    test_y = torch.load('../data/case_data_copy/test/test_y_outer21.pt')
    test_set1 = Data.TensorDataset(test_u.cuda(),
                                        test_y.cuda(),
                                      test_y.cuda())
    loader_outer07 = torch.utils.data.DataLoader(
        dataset=test_set1, batch_size=1, shuffle=False)
    target_list1 = []
    for idx, (batch_x,batch_obs, batch_y) in enumerate(loader_outer07):
        target_list1.append(batch_y.cpu())
    
    plt.figure(figsize=(8,4.5))
    plt.plot(target_list1[0].numpy().T)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Acceleration', fontsize=15)
    plt.grid(True)
    plt.savefig('../figs/splited_example', dpi=300, bbox_inches='tight')

    plt.show()

def step8():
    
    model = torch.load('../models/femto_15_5_serial_newdata.pt')
    test_u = torch.load('../data/femto_show_u.pt')
    test_y = torch.load('../data/femto_show_y.pt')
    testset = Data.TensorDataset(test_u.cuda(),
                                        test_y.cuda(),
                                      test_y.cuda())
    loader= torch.utils.data.DataLoader(
        dataset=testset, batch_size=1, shuffle=False)
    rus = []
    y = []
    for idx, (batch_x,batch_obs, batch_y) in enumerate(loader):
        pred = model(batch_x, y_obs=batch_obs)
        rus.append(pred)
        y.append(batch_y)
        print('rmse: ', criterion(pred, batch_obs))
        print('r2: ', r2_loss(pred, batch_obs))
    plt.figure(figsize=(8,4.5))
    plt.plot(rus[0].detach().cpu().numpy().T, color='blue')
    plt.plot(y[0].detach().cpu().numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    plt.savefig('../figs/femto0_val', dpi=300, bbox_inches='tight')
    plt.figure(figsize=(8,4.5))
    plt.plot(rus[1].detach().cpu().numpy().T, color='blue')
    plt.plot(y[1].detach().cpu().numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    # plt.savefig('../figs/femto1_val', dpi=300, bbox_inches='tight')
    plt.figure(figsize=(8,4.5))
    plt.plot(rus[2].detach().cpu().numpy().T, color='blue')
    plt.plot(y[2].detach().cpu().numpy().T, color='red')
    plt.legend(['Predicted signal', 'Actual signal'], loc="upper right")
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('Timesteps',fontsize=15)
    plt.ylabel('Normalized Acceleration',fontsize=15)
    # plt.savefig('../figs/femto2_val', dpi=300, bbox_inches='tight')

    plt.show()

def step9():
    pass




if __name__ == "__main__":
    step3()
