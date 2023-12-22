import sys
import os

home = os.environ["HOME"]
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from my_model.mingpt.model import GPTD, GPTConfig
from my_model.mingpt.optimization import AdamWeightDecay
from my_model.mingpt.trainer import Trainer, TrainerConfig
from my_model.mingpt.utils import sample
import matplotlib.pyplot as plt
import seaborn as sb  # 混合行列
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras import Model, Input, layers
from keras.models import load_model
import math

class CharDataset:

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        for _ in range(self.__len__()):
            i = np.random.randint(0, len(self.data) - (self.block_size + 1))
            chunk = self.data[i:i+self.block_size+1]
            dix = [self.stoi[s] for s in chunk]
            x = tf.convert_to_tensor(dix[:-1], dtype=tf.int32)
            y = tf.convert_to_tensor(dix[1:], dtype=tf.int32)
            yield x, y
    
    __call__ = __iter__

def main():

  # 445g
  csv_convex = np.loadtxt(
      f"{home}/data/convex.csv", delimiter=",", encoding="utf_8_sig", unpack=True
  )
  # 692g
  csv_cylinder = np.loadtxt(
      f"{home}/data/cylinder.csv", delimiter=",", encoding="utf_8_sig", unpack=True
  )
  # 1118g
  csv_wall = np.loadtxt(
      f"{home}/data/wall.csv", delimiter=",", encoding="utf_8_sig", unpack=True
  )


  # 時間の行を削除
  csv_convex = np.delete(csv_convex, 0, 0)
  csv_cylinder = np.delete(csv_cylinder, 0, 0)
  csv_wall = np.delete(csv_wall, 0, 0)

  # %%
  # データを格納、学習に使う長さを指定
  length_start = 1500
  length_end = 3000
  skip_num = 20

  data_convex = []
  data_cylinder = []
  data_wall = []

  # 入力値と教師データを格納
  for x in range(0, 10):
      for i in range(csv_convex.shape[0]):  # データの数
          tmp = csv_convex[i][length_start:length_end] + (0.1 * x)
          tmp = tmp[::skip_num]
          tmp = np.array2string(tmp)
          data_convex.append(tmp[1:-1] + " $" + " Y")
          data_cylinder.append(tmp[1:-1] + " $" + " N")
          data_wall.append(tmp[1:-1] + " $" + " N")
      for i in range(csv_cylinder.shape[0]):
          tmp = csv_convex[i][length_start:length_end] + (0.1 * x)
          tmp = tmp[::skip_num]
          tmp = np.array2string(tmp)
          data_convex.append(tmp[1:-1] + " $" + " N")
          data_cylinder.append(tmp[1:-1] + " $" + " Y")
          data_wall.append(tmp[1:-1] + " $" + " N")
      for i in range(csv_wall.shape[0]):
          tmp = csv_convex[i][length_start:length_end] + (0.1 * x)
          tmp = tmp[::skip_num]
          tmp = np.array2string(tmp)
          data_convex.append(tmp[1:-1] + " $" + " N")
          data_cylinder.append(tmp[1:-1] + " $" + " N")
          data_wall.append(tmp[1:-1] + " $" + " Y")

  block_size = 100
  train_dataset_gen_convex = CharDataset("".join(data_convex), block_size)
  train_dataset_convex = tf.data.Dataset.from_generator(train_dataset_gen_convex,(tf.int32,tf.int32))
  mconf_convex = GPTConfig(train_dataset_gen_convex.vocab_size, train_dataset_gen_convex.block_size, n_layer=1, n_head=8, n_embd=64)
  tconf_convex = TrainerConfig(max_epochs=10, batch_size=64, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset_gen_convex)*block_size,
                      num_workers=4)
  trainer_convex = Trainer(GPTD, mconf_convex, train_dataset_convex, len(train_dataset_gen_convex), None, None, tconf_convex)

  trainer_convex.train()

  context = data_convex[100][:-1]
  print(train_dataset_gen_convex.stoi)
  x = tf.convert_to_tensor([train_dataset_gen_convex.stoi[s] for s in context], dtype=tf.int32)[None,...]
  y = sample(trainer_convex.model, x, 1, temperature=0.9, sample=True, top_k=5)[0]
  completion = ''.join([train_dataset_gen_convex.itos[int(i)] for i in y])
  print(data_convex[100])
  print(completion)

if __name__ == '__main__':
  main()