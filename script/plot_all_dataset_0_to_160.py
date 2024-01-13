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
# 445g
csv_convex = np.loadtxt(f"{home}/data/convex.csv", delimiter=",", dtype="float",)
# 692g
csv_cylinder = np.loadtxt(f"{home}/data/cylinder.csv", delimiter=",", dtype="float",)
# 1118g
csv_wall = np.loadtxt(f"{home}/data/wall.csv", delimiter=",", dtype="float",)


plt.figure(figsize=(15,10),dpi=700)

# 軸ラベル設定
plt.xlabel("経過時間 [s]",size=30,labelpad=15,)
plt.ylabel("測定電圧 [V]",size=30,labelpad=15,)

# 軸設定
plt.xlim([0,0.4]) # 軸範囲
plt.ylim([0,60])
plt.xticks([0, 0.1, 0.2, 0.3, 0.4]) # 軸に表示させる数値

#グリッド線表示
plt.grid()

# x軸データ
x = csv_convex[:,0] * 0.0025

for i in range(0,10):
    for j in range(1, 11):
        if i == 0 and j == 10:
            plt.plot(x,csv_convex[:,j:]+ (0.1 * i),c='tab:red',lw=2.5,label='445g')
        else:
            plt.plot(x,csv_convex[:,j:]+ (0.1 * i),c='tab:red',lw=2.5,)
for i in range(0,10):
    for j in range(1, 11):
        if i == 0 and j == 10:
            plt.plot(x,csv_cylinder[:,j:]+ (0.1 * i),c='tab:blue',lw=2.5,label='692g')
        else:
            plt.plot(x,csv_cylinder[:,j:]+ (0.1 * i),c='tab:blue',lw=2.5,)
for i in range(0,10):
    for j in range(1, 11):
        if i == 0 and j == 10:
            plt.plot(x,csv_wall[:,j:]+ (0.1 * i),c='tab:green',lw=2.5,label='1118g')
        else:
            plt.plot(x,csv_wall[:,j:]+ (0.1 * i),c='tab:green',lw=2.5,)

#凡例の表示
plt.legend(fontsize=30)

#%%
# グラフの保存
plt.savefig(f"{home}/data/plot_all_dataset_0_to_160.png")
