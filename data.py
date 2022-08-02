import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import scipy.io as scio
from utils import import_data


class MyData:

    def __init__(self):
        pass

    @staticmethod
    def get_filter_data():
        x_train, x_test, y_train, y_test = import_data("../data/dataset/")
        label = np.load("../data/new_data/test/lowpass_test_2.npy")
        x_train = x_test

        # label.resize([11821, 576, 1])
        x_train.resize([x_train.shape[0], 576])

        x_train = torch.tensor(x_train, dtype=float)
        label = torch.tensor(label, dtype=float)
        torch.squeeze(x_train)
        print(x_train.shape, label.shape)

        # x_train = x_train.cuda(device)
        # label = label.cuda(device)

        dataset = Data.TensorDataset(x_train, label)

        return dataset

    @staticmethod
    def get_state_data():
        x_train = torch.load('../data/bearing_data/x_train.pt')
        y_train = torch.load('../data/bearing_data/y_train_ds.pt')

        dataset = Data.TensorDataset(x_train, y_train)

        return dataset

    @staticmethod
    def get_batterie_data():
        x_train = torch.load('../data/bearing_data/x_train_bt.pt')
        y_train = torch.load('../data/bearing_data/y_train_bt.pt')

        dataset = Data.TensorDataset(x_train, y_train)

        return dataset

    @staticmethod
    def get_self_made_data():
        path = '../data/dataset/train_sec_order.mat'
        data_dict = scio.loadmat(path)
        d_in = data_dict.get('u_all')
        d_out = data_dict.get('y_all')
        d_in = np.delete(d_in, 0, axis=0)
        d_out = np.delete(d_out, 0, axis=0)

        # d_init = d_out[:, 0]
        # dataset = Data.TensorDataset(torch.tensor(d_in), torch.tensor(d_init), torch.tensor(d_out))
        dataset = Data.TensorDataset(torch.tensor(d_in), torch.tensor(d_out), torch.tensor(d_out))

        return dataset

    @staticmethod
    def get_10hz_data():
        x_train = torch.load('../data/bearing_data/x_train_10hz.pt')
        y_train = torch.load('../data/bearing_data/y_train_10hz.pt')
        # 20 data
        x_train = torch.reshape(x_train, (20, 20, 2))
        x_train = torch.transpose(x_train, 1, 2)
        dataset = Data.TensorDataset(x_train, y_train, y_train)

        return dataset

    @staticmethod
    def get_time_variant_data():
        path = '../data/dataset/time_variant_data.mat'
        data_dict = scio.loadmat(path)
        d_in = data_dict.get("u")
        d_out = data_dict.get("y")
        dataset = Data.TensorDataset(torch.FloatTensor(d_in), torch.FloatTensor(d_out), torch.FloatTensor(d_out))

        return dataset

    @staticmethod
    def get_motor_data():
        data_dict = scio.loadmat('../data/dataset/train_motor.mat')
        d_in = data_dict.get("u")
        d_in = d_in[:, ::3]
        d_out = data_dict.get("y")
        d_out = d_out[:, ::3]
        dataset = Data.TensorDataset(torch.FloatTensor(d_in).cuda(),
                                     torch.FloatTensor(d_out).cuda(),
                                     torch.FloatTensor(d_out).cuda())

        return dataset

    @staticmethod
    def get_bearing_data():
        path = '../data/bearing_data/train_outer_1_1.mat'
        data_dict = scio.loadmat(path)
        d_out = data_dict.get('acc_x_outer_1_1_mat')
        d_in = np.zeros_like(d_out)
        dataset = Data.TensorDataset(torch.FloatTensor(d_in).cuda(),
                                     torch.FloatTensor(d_out).cuda(),
                                     torch.FloatTensor(d_out).cuda())

        return dataset

    @staticmethod
    def get_outer_data():
        outer_u = torch.load('../data/bearing_data/xjtu/u_outer.pt')
        outer_y = torch.load('../data/bearing_data/xjtu/y_outer.pt')
        dataset = Data.TensorDataset(outer_u.cuda(),
                                     outer_y.cuda(),
                                     outer_y.cuda())
        # length = 20
        return dataset

    @staticmethod
    def get_inner_data():
        inner_u = torch.load('../data/bearing_data/xjtu/inner_u.pt')
        inner_y = torch.load('../data/bearing_data/xjtu/inner_y.pt')
        dataset = Data.TensorDataset(inner_u.cuda(),
                                     inner_y.cuda(),
                                     inner_y.cuda())

        return dataset

    @staticmethod
    def get_inner_data_two():
        inner_u = torch.load('../data/bearing_data/xjtu/inner_u_2.pt')
        inner_y = torch.load('../data/bearing_data/xjtu/inner_y_2.pt')
        dataset = Data.TensorDataset(inner_u.cuda(),
                                     inner_y.cuda(),
                                     inner_y.cuda())

        return dataset

    @staticmethod
    def get_case_data(name:str):
        len_dict = {'fd07_outer':25,
                    'fd21_outer':41,
                    'fd07_inner':52,
                    'fd14_inner':39,
                    'fd21_inner':63,
                    'fd28_inner':56
                    }
        length = len_dict.get(name)
        train_u_name = '../data/case_data/train/train_u_' + name + '.pt'
        train_y_name = '../data/case_data/train/train_y_' + name + '.pt'
        test_u_name = '../data/case_data/test/test_u_' + name + '.pt'
        test_y_name = '../data/case_data/test/test_y_' + name + '.pt'
        train_u = torch.load(train_u_name)
        train_y = torch.load(train_y_name)
        test_u = torch.load(test_u_name)
        test_y = torch.load(test_y_name)
        train_set = Data.TensorDataset(train_u.cuda(),
                                        train_y.cuda(),
                                        train_y.cuda())
        test_set = Data.TensorDataset(test_u.cuda(),
                                        test_y.cuda(),
                                        test_y.cuda())

        return train_set, test_set, length