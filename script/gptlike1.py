import sys
import os

home = os.environ["HOME"]
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split
import numpy as np
from my_model.mingpt.model import GPT, GPTConfig, TransformerClassifier
from my_model.mingpt.optimization import AdamWeightDecay


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
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])
      for i in range(csv_cylinder.shape[0]):
          tmp = csv_cylinder[i][length_start:length_end] + (0.1 * x)
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])
      for i in range(csv_wall.shape[0]):
          tmp = csv_wall[i][length_start:length_end] + (0.1 * x)
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])

  x_convex = np.array(data_convex).reshape(len(data_convex), int((length_end - length_start) / skip_num), 1)

  # 訓練データ、検証データ、テストデータに分割
  x_convex_train, x_convex_test = train_test_split(
      x_convex, test_size=int(len(data_convex) * 0.4)
  )
  x_convex_valid, x_convex_test = train_test_split(
      x_convex_test, test_size=int(len(x_convex_test) * 0.5)
  )

  x_cylinder = np.array(data_cylinder).reshape(len(data_cylinder), int((length_end - length_start) / skip_num), 1)

  # 訓練データ、検証データ、テストデータに分割
  x_cylinder_train, x_cylinder_test = train_test_split(
      x_cylinder, test_size=int(len(data_cylinder) * 0.4)
  )
  x_cylinder_valid, x_cylinder_test = train_test_split(
      x_cylinder_test, test_size=int(len(x_cylinder_test) * 0.5)
  )

  x_wall = np.array(data_wall).reshape(len(data_wall), int((length_end - length_start) / skip_num), 1)

  # 訓練データ、検証データ、テストデータに分割
  x_wall_train, x_wall_test = train_test_split(
      x_wall, test_size=int(len(data_wall) * 0.4)
  )
  x_wall_valid, x_wall_test = train_test_split(
      x_wall_test, test_size=int(len(x_wall_test) * 0.5)
  )

  m_conf = GPTConfig(int((length_end - length_start) / skip_num), int((length_end - length_start) / skip_num), n_layer=1, n_head=8, n_embd=64)
  model_convex = GPT(m_conf,(x_convex_train.shape[1],))
  model_cylinder = GPT(m_conf,(x_cylinder_train.shape[1],))
  model_wall = GPT(m_conf,(x_wall_train.shape[1],))

  learning_rate = 3e-4
  betas = (0.9, 0.95)
  grad_norm_clip = 1.0
  weight_decay = 0.1  # only applied on matmul weights
  optimizer = AdamWeightDecay(learning_rate=learning_rate, # type: ignore
                                              weight_decay_rate=weight_decay,
                                              beta_1=betas[0], beta_2=betas[1],
                                              gradient_clip_norm=grad_norm_clip,
                                              exclude_from_weight_decay=['layer_normalization', 'bias'])
  model_convex.compile(
      optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"]
  )
  model_cylinder.compile(
      optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"]
  )
  model_wall.compile(
      optimizer=optimizer, loss="mean_squared_error", metrics=["accuracy"]
  )

  epochs = 1
  batch_size = 1

  model_convex.fit(
      x_convex_train,
      x_convex_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_convex_valid, x_convex_valid),
  )

  model_cylinder.fit(
      x_cylinder_train,
      x_cylinder_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_cylinder_valid, x_cylinder_valid),
  )

  model_wall.fit(
      x_wall_train,
      x_wall_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_wall_valid, x_wall_valid),
  )

  model_convex.save("model_convex")
  model_cylinder.save("model_cylinder")
  model_wall.save("model_wall")

if __name__ == '__main__':
  main()