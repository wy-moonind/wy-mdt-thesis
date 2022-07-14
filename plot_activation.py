from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

# case 07 data
df = pd.read_csv('../case_21.csv', header=None, sep=',')
df = np.array(df)
print(df.shape)
train_org = df[:5, :7]
val_org = df[:5, 7:14]
r2_org = df[:5, 14:]

train_tanh = df[6:11, :7]
val_tanh = df[6:11, 7:14]
r2_tanh = df[6:11, 14:]

train_relu = df[12, :7]
val_relu = df[12, 7:14]
r2_relu = df[12, 14:]

train_softplus = df[14, :7]
val_softplus = df[14, 7:14]
r2_softplus = df[14, 14:]

layer = [1, 2, 3, 4, 5]
order = [3, 5, 7, 9, 11, 13, 15]

# case 07 tanh single
plt.figure(0)
# train loss
plt.plot(order, train_org[0,:])
plt.plot(order, train_tanh[0,:])
plt.plot(order, train_relu[:])
plt.plot(order, train_softplus[:])
plt.legend(['None', 'Tanh', 'ReLU', 'Softplus'])
plt.ylim(0.1, 0.4)
plt.title('Different activation for single layer (train loss)')
plt.xlabel('Order')
plt.ylabel('RMSE loss')
plt.grid(True)
plt.savefig('../figs/activation/single_train_21', dpi=300)

plt.figure(1)
# val loss
plt.plot(order, val_org[0,:])
plt.plot(order, val_tanh[0,:])
plt.plot(order, val_relu[:])
plt.plot(order, val_softplus[:])
plt.legend(['None', 'Tanh', 'ReLU', 'Softplus'])
plt.ylim(0.1, 0.4)
plt.title('Different activation for single layer (validation loss)')
plt.xlabel('Order')
plt.ylabel('RMSE loss')
plt.grid(True)
plt.savefig('../figs/activation/single_val_21', dpi=300)

plt.figure(2)
# val r2
plt.plot(order, r2_org[0,:])
plt.plot(order, r2_tanh[0,:])
plt.plot(order, r2_relu[:])
plt.plot(order, r2_softplus[:])
plt.legend(['None', 'Tanh', 'ReLU', 'Softplus'])
plt.ylim(0, 0.7)
plt.title('Different activation for single layer (validation R2)')
plt.xlabel('Order')
plt.ylabel('R2 value')
plt.grid(True)
plt.savefig('../figs/activation/single_r2_21', dpi=300)

# plt.show()
