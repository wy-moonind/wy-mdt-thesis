import numpy as np
import scipy.io as scio
import datetime


def to_cat(data, num_classes=None):
    """
    Change the label to One-hot coding.
    :param data: label to change as in [1, 2, 3, 4]
    :param num_classes: total numbers of classes
    :return: Encoded label
    """
    # Each data should be represents the class number of the data
    if num_classes is None:
        num_classes = np.unique(data)
    data_class = np.zeros((data.shape[0], num_classes))
    for i in range(data_class.shape[0]):
        num = data[i]
        data_class[i, num] = 1
    return data_class


def import_data(path):
    """
    Import data from given path.
    :param path: string, Folder of the data files
    :param model: which model do we use
    :return: ndarray, training and test data
    """
    # import dataset
    x_train_file = path + 'traindata.mat'
    x_train_dict = scio.loadmat(x_train_file)
    x_train_all = x_train_dict.get('train_data')

    y_train_file = path + 'trainlabel.mat'
    y_train_dict = scio.loadmat(y_train_file)
    y_train_all = y_train_dict.get('train_label')

    x_test_file = path + 'testdata.mat'
    x_test_dict = scio.loadmat(x_test_file)
    x_test_all = x_test_dict.get('test_data')

    y_test_file = path + 'testlabel.mat'
    y_test_dict = scio.loadmat(y_test_file)
    y_test_all = y_test_dict.get('test_label')
    # loading complete
    # how many classes
    classes = np.unique(y_train_all)
    # x_train_all, x_vali, y_train_all, y_vali = train_test_split(x_train_all, y_train_all, test_size=0.9)
    size_train = x_train_all.shape[0]
    size_test = x_test_all.shape[0]
    x_train = x_train_all.reshape(size_train, 576, 1)
    x_test = x_test_all.reshape(size_test, 576, 1)
    y_train = to_cat(y_train_all, num_classes=4)
    y_test = to_cat(y_test_all, num_classes=4)

    return x_train, x_test, y_train, y_test


def cal_time():
    """
    Calculate code running time.

    :return: None
    """
    start = datetime.datetime.now()
    # do something
    end = datetime.datetime.now()
    print("time for backwarding: ", (end - start).microseconds)

def translate_params(params, candidate):
    order_candidate = candidate[0]
    layer_candidate = candidate[1]
    layer, order = None, None
    for j in range(len(order_candidate)):
        if (j / len(order_candidate)) <= params[0] <= ((j + 1) / len(order_candidate)):
            order = order_candidate[j]
        elif params[0] == 1:
            order = order_candidate[-1]
    for i in range(len(layer_candidate)):
        if (i / len(layer_candidate)) <= params[1] <= ((i + 1) / len(layer_candidate)):
            layer = layer_candidate[i]
        elif params[1] == 1:
            layer = layer_candidate[-1]
    assert layer in layer_candidate
    assert order in order_candidate

    return order, layer
