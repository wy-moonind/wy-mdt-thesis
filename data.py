
import torch
import torch.utils.data as Data


class MyData:

    def __init__(self):
        pass

    @staticmethod
    def get_case_data(name: str):
        """
        Get the CWRU dataset

        :param name: string, name of the sub dataset
        :return: torch.utils.data.TensorDataset, train_set, test_set, show_set selected from the test_set
        """
        train_u_name = '../data/case_data/train/train_u_' + name + '.pt'
        train_y_name = '../data/case_data/train/train_y_' + name + '.pt'
        test_u_name = '../data/case_data/test/test_u_' + name + '.pt'
        test_y_name = '../data/case_data/test/test_y_' + name + '.pt'
        show_u_name = '../data/case_data/test/show_u_' + name + '.pt'
        show_y_name = '../data/case_data/test/show_y_' + name + '.pt'
        train_u = torch.load(train_u_name)
        train_y = torch.load(train_y_name)
        test_u = torch.load(test_u_name)
        test_y = torch.load(test_y_name)
        show_u = torch.load(show_u_name)
        show_y = torch.load(show_y_name)
        train_set = Data.TensorDataset(train_u.cuda(),
                                       train_y.cuda(),
                                       train_y.cuda())
        test_set = Data.TensorDataset(test_u.cuda(),
                                      test_y.cuda(),
                                      test_y.cuda())
        show_set = Data.TensorDataset(show_u.cuda(),
                                      show_y.cuda(),
                                      show_y.cuda())
        return train_set, test_set, show_set

    @staticmethod
    def get_femto_data():
        """
        Get the CWRU dataset

        :return: torch.utils.data.TensorDataset, train_set, test_set, show_set selected from the test_set
        """
        train_u_path = '../data/femto_data/train/train_u_all.pt'
        train_y_path = '../data/femto_data/train/train_y_all.pt'
        test_u_path = '../data/femto_data/test/test_u_all.pt'
        test_y_path = '../data/femto_data/test/test_y_all.pt'
        show_u_path = '../data/femto_data/test/show_u_all.pt'
        show_y_path = '../data/femto_data/test/show_y_all.pt'
        train_u = torch.load(train_u_path)
        train_y = torch.load(train_y_path)
        test_u = torch.load(test_u_path)
        test_y = torch.load(test_y_path)
        show_u = torch.load(show_u_path)
        show_y = torch.load(show_y_path)
        train_set = Data.TensorDataset(train_u.cuda(),
                                       train_y.cuda(),
                                       train_y.cuda())
        test_set = Data.TensorDataset(test_u.cuda(),
                                      test_y.cuda(),
                                      test_y.cuda())
        show_set = Data.TensorDataset(show_u.cuda(), 
                                      show_y.cuda(), 
                                      show_y.cuda())
        return train_set, test_set, show_set
