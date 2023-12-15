from pyexpat import model
import sys
import os

from py import test

home = os.environ['HOME'];
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split 
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from my_model.mingpt.model import GPT, GPT1Config, GPTConfig 
from my_model.mingpt.trainer import TrainerConfig, Trainer
from my_model.mingpt.utils import sample


class Dataset:

    def __init__(self, data):
        vocab_size = 100 # 3種類の錘
        length_start = 1500
        length_end = 3000
        
        self.vocab_size = vocab_size
        self.block_size = (length_end - length_start)
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()): 
          item = self.data[i]
          x = item[:-1]
          y = item[1:]

          x = tf.convert_to_tensor(x,dtype=tf.float32)
          y = tf.convert_to_tensor(y,dtype=tf.float32)

          yield x,y

    __call__ = __iter__


# csv_1 = np.loadtxt(f"{home}/data/isodisplacement1.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_2 = np.loadtxt(f"{home}/data/isodisplacement2.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_3 = np.loadtxt(f"{home}/data/isodisplacement3.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_4 = np.loadtxt(f"{home}/data/isodisplacement4.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_5 = np.loadtxt(f"{home}/data/isodisplacement5.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_6 = np.loadtxt(f"{home}/data/isodisplacement6.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_7 = np.loadtxt(f"{home}/data/isodisplacement7.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_8 = np.loadtxt(f"{home}/data/isodisplacement8.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_9 = np.loadtxt(f"{home}/data/isodisplacement9.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])
# csv_10 = np.loadtxt(f"{home}/data/isodisplacement10.csv", delimiter=",", encoding="utf_8_sig", unpack=True, skiprows=3, usecols=[3,8,13,18,23,28,33,38,43,48])

#445g
csv_convex = np.loadtxt(f"{home}/data/convex.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
#692g
csv_cylinder = np.loadtxt(f"{home}/data/cylinder.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
#1118g
csv_wall = np.loadtxt(f"{home}/data/wall.csv", delimiter=",", encoding="utf_8_sig", unpack=True)



# 時間の行を削除
csv_convex = np.delete(csv_convex, 0, 0)
csv_cylinder = np.delete(csv_cylinder, 0, 0)
csv_wall = np.delete(csv_wall, 0, 0)

# %%
# データを格納、学習に使う長さを指定
length_start = 1500
length_end = 3000

data = []  # 入力値 + 教師データ

# 入力値と教師データを格納
for i in range(csv_convex.shape[0]):  # データの数
    tmp = csv_convex[i][length_start:length_end]
    tmp = tmp[::2] #データ数を半分に
    tmp = np.append(tmp, 0) #教師データ追加

    data.append(tmp) 
for i in range(csv_cylinder.shape[0]):
    tmp = csv_cylinder[i][length_start:length_end]
    tmp = tmp[::2] #データ数を半分に
    tmp = np.append(tmp, 1)  #教師データ追加
    data.append(tmp) 
for i in range(csv_wall.shape[0]):
    tmp = csv_wall[i][length_start:length_end]
    tmp = tmp[::2] #データ数を半分に
    tmp = np.append(tmp, 2)  #教師データ追加
    data.append(tmp) 


# 訓練データ、テストデータに分割
x_train, x_test = train_test_split(
    data, test_size=int(len(data) * 0.4), 
)

train_dataset_gen = Dataset(x_train)
test_dataset_gen = Dataset(x_test)

train_dataset = tf.data.Dataset.from_generator(train_dataset_gen,(tf.float32,tf.float32))
test_dataset = tf.data.Dataset.from_generator(test_dataset_gen,(tf.float32,tf.float32))

m_conf = GPTConfig(train_dataset_gen.vocab_size, train_dataset_gen.block_size, 
                  n_layer=2, n_head=4, n_embd=128)
t_conf = TrainerConfig(max_epochs=5, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset_gen)*(4),
                      num_workers=4)

trainer = Trainer(model=GPT,model_config=m_conf,train_dataset=train_dataset,train_dataset_len=len(train_dataset_gen),test_dataset=test_dataset, test_dataset_len=len(test_dataset_gen), config=t_conf)
trainer.train()

print("test_start")
loader = train_dataset.batch(512)
for b, (x, y) in enumerate(loader):
  result = sample(trainer.model,x,1,sample=True)
  print("result:")
  print(result)