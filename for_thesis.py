import scipy.io as scio
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

path = 'C:/Users/wangy/Desktop/Thesis_new/bearing_data/case/outer_fd21/246.mat'
data_dict = scio.loadmat(path)
data = data_dict.get('X246_DE_time')
print(data.shape)

plt.figure(figsize=(12,4))
plt.plot(data[:10000])
plt.title('Fault peak selection and segmentation(CWRU fault diameter=0.021”)')
plt.ylabel('Accleration')
plt.xlabel('Timestep')
plt.grid(True)
# plt.show()
plt.savefig('C:/Users/wangy/Desktop/Thesis_new/figures/fault_split', dpi=300)

# def get_vibration(path: str):
#     df = pd.read_csv(path, header=None, sep=',')
#     df_np = df.to_numpy()
#     vib = df_np[:, 4]
#     return vib


# df = pd.read_csv('C:/Users/wangy/Desktop/Thesis_new/phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_02752.csv', header=None, sep=",")

# vib1 = get_vibration('C:/Users/wangy/Desktop/Thesis_new/phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_00001.csv')
# vib2 = get_vibration('C:/Users/wangy/Desktop/Thesis_new/phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_00701.csv')
# vib3 = get_vibration('C:/Users/wangy/Desktop/Thesis_new/phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_01601.csv')
# vib4 = get_vibration('C:/Users/wangy/Desktop/Thesis_new/phm-ieee-2012-data-challenge-dataset-master/Learning_set/Bearing1_1/acc_02752.csv')

# print(vib1.shape)

# plt.figure(figsize=(8,6))
# plt.subplot(3, 1, 1)
# plt.plot(vib1)
# plt.ylabel('Vibration')
# plt.subplot(3, 1, 2)
# plt.plot(vib3)
# plt.ylabel('Vibration')
# plt.subplot(3, 1, 3)
# plt.plot(vib4)
# plt.ylabel('Vibration')

# plt.savefig('C:/Users/wangy/Desktop/Thesis_new/figures/femto_example', dpi=300)

# plt.show()
