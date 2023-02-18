import scipy.io as scio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

path = '../../bearing_data/case/outer_fd21/246.mat'
data_dict = scio.loadmat(path)
data = data_dict.get('X246_DE_time')
print(data.shape)

plt.figure(figsize=(12,5))
plt.plot(data[:10000])
plt.title('Fault peak selection and segmentation(CWRU fault diameter=0.021”)', fontsize=15)
plt.ylabel('Acceleration $[m/s^2]$', fontsize=15)
plt.xlabel('Timestep', fontsize=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.grid(True)
# plt.show()
plt.savefig('../../figures/fault_split_new', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.plot(data[:10000])
plt.title('CWRU data example (outer race, fd = 0.021")', fontsize=15)
plt.ylabel('Acceleration $[m/s^2]$', fontsize=15)
plt.xlabel('Timestep', fontsize=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.grid(True)
# plt.show()
plt.savefig('../../figures/cwru_example_new', dpi=300, bbox_inches='tight')

# def get_vibration(path: str):
#     df = pd.read_csv(path, header=None, sep=',')
#     df_np = df.to_numpy()
#     vib = df_np[:, 4]
#     return vib


# df = pd.read_csv('../../femto/Learning_set/Bearing1_1/acc_02752.csv', header=None, sep=",")

# vib1 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_00001.csv')
# vib2 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_00701.csv')
# vib3 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_01601.csv')
# vib4 = get_vibration('../../femto/Learning_set/Bearing1_1/acc_02752.csv')

# print(vib1.shape)

# plt.figure(figsize=(10,8))
# plt.subplot(3, 1, 1)
# plt.plot(vib1)
# plt.ylabel('Vibration', fontsize=15)
# plt.legend(['First FEMTO dataset example'], loc='upper right', fontsize=15)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.subplot(3, 1, 2)
# plt.plot(vib3)
# plt.ylabel('Vibration', fontsize=15)
# plt.legend(['Second FEMTO dataset example'], loc='upper right', fontsize=15)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.subplot(3, 1, 3)
# plt.plot(vib4)
# plt.ylabel('Vibration', fontsize=15)
# plt.legend(['Third FEMTO dataset example'], loc='upper right', fontsize=15)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.xlabel('Timestep', fontsize=15)

# # plt.savefig('C:/Users/wangy/Desktop/Thesis_new/figures/femto_example', dpi=300)

# plt.show()
