import scipy.io as scio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch


def cwru_example():
    path = '../../bearing_data/case/outer_fd21/246.mat'
    data_dict = scio.loadmat(path)
    data = data_dict.get('X246_DE_time')
    plt.figure(figsize=(10,6))
    plt.plot(data[:10000])
    plt.title('CWRU data example (outer race, fd = 0.021")', fontsize=15)
    plt.ylabel('Accleration $[m/s^2]$', fontsize=15)
    plt.xlabel('Timestep', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid(True)
    # plt.show()
    plt.savefig('../../figures/cwru_example_new', dpi=300, bbox_inches='tight')

def femto_example():
    def get_vibration(path: str):
        df = pd.read_csv(path, header=None, sep=',')
        df_np = df.to_numpy()
        vib = df_np[:, 4]
        return vib

    df = pd.read_csv('../../femto/Learning_set/Bearing1_1/acc_02752.csv', header=None, sep=",")

    vib1 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_00001.csv')
    # vib2 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_00701.csv')
    vib3 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_01601.csv')
    vib4 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_02752.csv')

    plt.figure(figsize=(10,8))
    plt.subplot(3, 1, 1)
    plt.plot(vib1)
    plt.ylabel('Accleration $[m/s^2]$', fontsize=15)
    plt.title('(a)')
    plt.legend(['First FEMTO dataset example'], loc='upper right', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.subplot(3, 1, 2)
    plt.plot(vib3)
    plt.ylabel('Accleration $[m/s^2]$', fontsize=15)
    plt.title('(b)')
    plt.legend(['Second FEMTO dataset example'], loc='upper right', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.subplot(3, 1, 3)
    plt.plot(vib4)
    plt.ylabel('Accleration $[m/s^2]$', fontsize=15)
    plt.title('(c)')
    plt.legend(['Second FEMTO dataset example'], loc='upper right', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('Timestep', fontsize=15)
    plt.tight_layout()
    plt.savefig('../../figures/femto_example_new', dpi=300)
    plt.show()

def fault_split():
    path = '../../bearing_data/case/outer_fd21/246.mat'
    data_dict = scio.loadmat(path)
    data = data_dict.get('X246_DE_time')
    plt.figure(figsize=(12,5))
    plt.plot(data[:10000])
    plt.title('Fault peak selection and segmentation(CWRU fault diameter=0.021”)', fontsize=15)
    plt.ylabel('Accleration $[m/s^2]$', fontsize=15)
    plt.xlabel('Timestep', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid(True)
    # plt.show()
    plt.savefig('../../figures/fault_split_new', dpi=300, bbox_inches='tight')

def envelope():
    case_all = scio.loadmat('../../bearing_data/case/outer_fd21/246.mat')
    y246 = case_all.get('X246_DE_time')
    y = y246[0, :]
    t = np.linspace(0, 1, num=y.shape[0]) / 12000

    plt.figure(figsize=(10,6))
    envelope = scio.loadmat('../../bearing_data/case/envelope.mat')
    es = envelope.get('es')
    f = envelope.get('f')
    plt.plot(f, es)
    plt.vlines(107.5, 0, 0.07, colors='r', linestyles='dashed')
    plt.vlines(107.5 * 2, 0, 0.07, colors='r', linestyles='dashed')
    plt.vlines(107.5 * 3, 0, 0.07, colors='r', linestyles='dashed')
    plt.vlines(107.5 * 4, 0, 0.07, colors='r', linestyles='dashed')
    plt.legend(["Envelope spectrum of the signal", "Characteristic frequency"], fontsize=15)
    plt.xlabel('Frequency [Hz]', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title('Envelope spectrum of outer race data (fd = 0.021")', fontsize=15)
    plt.xlim(0, 500)
    plt.ylim(0, 0.07)
    plt.grid(True)
    plt.savefig('../../figures/envelope_new', dpi=300, bbox_inches='tight')

    plt.show()

def data_af_seg():
    data = torch.load('../data/case_data/test/test_y_outer21.pt')
    print(data.shape)
    plt.figure(figsize=(8,4))
    plt.plot(data[0,:])
    plt.xlabel('Timestep', fontsize=15)
    plt.ylabel('Normalized Accleration', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../figures/splited_example_new', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # cwru_example()
    femto_example()
    # fault_split()
    # envelope()
    # data_af_seg()
