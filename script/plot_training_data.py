#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

home = os.environ["HOME"]

def task(num):
  #%%
  #全体のパラメータ
  plt.rcParams['font.family'] = 'Noto Sans JP'
  plt.rcParams["xtick.labelsize"] = 25 # x軸のフォントサイズ
  plt.rcParams["ytick.labelsize"] = 25  # y軸のフォントサイズ
  #%%
  # データ読み込み、サイズ設定
  training_data = np.loadtxt(f"{home}/data/training_{num}.csv", delimiter=",", dtype="float",skiprows=1)

  fig, ax = plt.subplots(figsize=(30,10),dpi=700)
  twin1 = ax.twinx()
  twin2 = ax.twinx()
  twin3 = ax.twinx()


  epoch = training_data[:,0]
  accuracy = training_data[:,1]
  loss = training_data[:,2]
  v_accuracy = training_data[:,3]
  v_loss = training_data[:,4]

  p1, = ax.plot(epoch, accuracy, 'C0', label='train_accuracy')
  p2, = twin1.plot(epoch, loss, 'C1', label='train_loss')
  p3, = twin1.plot(epoch, v_accuracy, 'C2', label='valid_accuracy')
  p4, = twin3.plot(epoch, v_loss, 'C3', label='valid_loss')

  ax.set_xlabel('epoch数',size=30,labelpad=15,)
  ax.set_ylabel('train_accuracy',size=30,labelpad=15,)
  twin1.set_ylabel('train_loss',size=30,labelpad=15,)
  twin2.set_ylabel('valid_accuracy',size=30,labelpad=15,)
  twin3.set_ylabel('valid_loss',size=30,labelpad=15,)

  twin2.spines["right"].set_position(("axes", 1.2))
  twin3.spines["right"].set_position(("axes", 1.4))

  #凡例の表示
  plt.legend(fontsize=30, handles=[p1, p2, p3, p4])

  #%%
  # グラフの保存
  plt.savefig(f"{home}/data/training_data_{num}.png")

def main():  
  num = sys.argv[1]
  task(num=num)

if __name__ == "__main__":
    main()