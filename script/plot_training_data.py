#%%
import numpy as np
import matplotlib.pyplot as plt
import os

home = os.environ["HOME"]

#%%
#全体のパラメータ
plt.rcParams['font.family'] = 'Noto Sans JP'
plt.rcParams["xtick.labelsize"] = 25 # x軸のフォントサイズ
plt.rcParams["ytick.labelsize"] = 25  # y軸のフォントサイズ
#%%
# データ読み込み、サイズ設定
training_data = np.loadtxt(f"{home}/training.csv", delimiter=",", dtype="float",skiprows=1)

fig, ax = plt.subplots(figsize=(15,10),dpi=700)
twin1 = ax.twinx()

epoch = training_data[:,0]
accuracy = training_data[:,1]
loss = training_data[:,2]

p1, = ax.plot(epoch, accuracy, 'C0', label='accuracy')
p2, = twin1.plot(epoch, loss, 'C1', label='loss')

ax.set_xlabel('epoch数',size=30,labelpad=15,)
ax.set_ylabel('accuracy',size=30,labelpad=15,)
twin1.set_ylabel('loss',size=30,labelpad=15,)
#凡例の表示
plt.legend(fontsize=30, handles=[p1, p2])

#%%
# グラフの保存
plt.savefig(f"{home}/data/training_data.png")
