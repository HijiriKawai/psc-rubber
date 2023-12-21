import sys
import os

home = os.environ["HOME"]
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from my_model.mingpt.model import GPT, GPTConfig
from my_model.mingpt.optimization import AdamWeightDecay
import matplotlib.pyplot as plt
import seaborn as sb  # 混合行列
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras import Model, Input, layers


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
  skip_num = 2

  data = []  # 入力値
  data_convex = []
  data_cylinder = []
  data_wall = []
  target = []  # 教師データ
  target_convex = []
  target_cylinder = []
  target_wall = []

  # 入力値と教師データを格納
  for x in range(0, 10):
      for i in range(csv_convex.shape[0]):  # データの数
          tmp = csv_convex[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])  # データ数を半分にしながら挿入
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])
          target.append(0)
          target_convex.append(0)
          target_cylinder.append(1)
          target_wall.append(1)
      for i in range(csv_cylinder.shape[0]):
          tmp = csv_cylinder[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])
          target.append(1)
          target_convex.append(1)
          target_cylinder.append(0)
          target_wall.append(1)
      for i in range(csv_wall.shape[0]):
          tmp = csv_wall[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])
          data_convex.append(tmp[::skip_num])
          data_cylinder.append(tmp[::skip_num])
          data_wall.append(tmp[::skip_num])
          target.append(2)
          target_convex.append(1)
          target_cylinder.append(1)
          target_wall.append(0)


  # 訓練データ、テストデータに分割
  x = np.array(data).reshape(len(data), int((length_end - length_start) / skip_num), 1)
  t = np.array(target).reshape(len(target), 1)
  t = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

  # 訓練データ、検証データ、テストデータに分割
  x_train, x_test, t_train, t_test = train_test_split(
      x, t, test_size=int(len(data) * 0.4), stratify=t
  )
  x_valid, x_test, t_valid, t_test = train_test_split(
      x_test, t_test, test_size=int(len(x_test) * 0.5), stratify=t_test
  )

  x_convex = np.array(data_convex).reshape(len(data_convex), int((length_end - length_start) / skip_num), 1)
  t_convex = np.array(target_convex).reshape(len(target_convex), 1)
  t_convex = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

  # 訓練データ、検証データ、テストデータに分割
  x_convex_train, x_convex_test, t_convex_train, t_convex_test = train_test_split(
      x_convex, t_convex, test_size=int(len(data_convex) * 0.4), stratify=t_convex
  )
  x_convex_valid, x_convex_test, t_convex_valid, t_convex_test = train_test_split(
      x_convex_test, t_convex_test, test_size=int(len(x_convex_test) * 0.5), stratify=t_convex_test
  )

  x_cylinder = np.array(data_cylinder).reshape(len(data_cylinder), int((length_end - length_start) / skip_num), 1)
  t_cylinder = np.array(target_cylinder).reshape(len(target_cylinder), 1)
  t_cylinder = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

  # 訓練データ、検証データ、テストデータに分割
  x_cylinder_train, x_cylinder_test, t_cylinder_train, t_cylinder_test = train_test_split(
      x_cylinder, t_cylinder, test_size=int(len(data_cylinder) * 0.4), stratify=t_cylinder
  )
  x_cylinder_valid, x_cylinder_test, t_cylinder_valid, t_cylinder_test = train_test_split(
      x_cylinder_test, t_cylinder_test, test_size=int(len(x_cylinder_test) * 0.5), stratify=t_cylinder_test
  )

  x_wall = np.array(data_wall).reshape(len(data_wall), int((length_end - length_start) / skip_num), 1)
  t_wall = np.array(target_wall).reshape(len(target_wall), 1)
  t_wall = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

  # 訓練データ、検証データ、テストデータに分割
  x_wall_train, x_wall_test, t_wall_train, t_wall_test = train_test_split(
      x_wall, t_wall, test_size=int(len(data_wall) * 0.4), stratify=t_wall
  )
  x_wall_valid, x_wall_test, t_wall_valid, t_wall_test = train_test_split(
      x_wall_test, t_wall_test, test_size=int(len(x_test) * 0.5), stratify=t_wall_test
  )

  m_conf = GPTConfig(int((length_end - length_start) / skip_num), int((length_end - length_start) / skip_num), n_layer=12, n_head=8, n_embd=64)
  model_convex = GPT(m_conf)
  model_cylinder = GPT(m_conf)
  model_wall = GPT(m_conf)

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
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )
  model_cylinder.compile(
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )
  model_wall.compile(
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )

  epochs = 50
  batch_size = 10

  model_convex.fit(
      x_convex_train,
      t_convex_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_convex_valid, t_convex_valid),
  )

  model_cylinder.fit(
      x_cylinder_train,
      t_cylinder_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_cylinder_valid, t_cylinder_valid),
  )

  model_wall.fit(
      x_wall_train,
      t_wall_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_wall_valid, t_wall_valid),
  )

  model_convex = layers.Dense(1, activation="linear")(model_convex)
  model_cylinder = layers.Dense(1, activation="linear")(model_cylinder)
  model_wall = layers.Dense(1, activation="linear")(model_wall)
  concatenated = layers.concatenate([model_convex, model_cylinder, model_wall])
  result = layers.Dense(3, activation="softmax")(concatenated)
  model = Model(inputs=[Input(model_convex.input_shape[1 :])], outputs=[result])

  result = model.fit(
      x_train,
      t_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_valid, t_valid),
  )

  # %%
  # 正解率の可視化
  plt.figure(dpi=700)
  plt.plot(range(1, epochs + 1), result.history["accuracy"], label="train_acc")  # type: ignore
  plt.plot(range(1, epochs + 1), result.history["val_accuracy"], label="valid_acc")  # type: ignore
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.savefig(f"{home}/result/transformer-classifier/transformer_classifier_accuracy.png")
  # %%
  # 損失関数の可視化
  plt.figure(dpi=700)
  plt.plot(range(1, epochs + 1), result.history["loss"], label="training_loss")  # type: ignore
  plt.plot(range(1, epochs + 1), result.history["val_loss"], label="validation_loss")  # type: ignore
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.savefig(f"{home}/result/transformer-classifier/transformer_classifier_loss.png")
  # %%
  # 学習モデルを用いてx_trainから予測
  score_train = model.predict(x_train)

  # 学習モデルを用いてx_testから予測
  score_test = model.predict(x_test)

  # 正解率を求める
  count_train = 0
  count_test = 0

  for i in range(len(score_train)):
      if np.argmax(score_train[i]) == np.argmax(t_train[i]):
          count_train += 1

  for i in range(len(score_test)):
      if np.argmax(score_test[i]) == np.argmax(t_test[i]):
          count_test += 1

  print(epochs, batch_size)
  print("train_acc=")
  print(count_train / len(score_train))
  print("test_acc=")
  print(count_test / len(score_test))


  # %%
  # 混合行列生成の関数
  def print_mtrix(t_true, t_predict):
      mtrix_data = confusion_matrix(t_true, t_predict)
      df_mtrix = pd.DataFrame(
          mtrix_data,
          index=["445g", "692g", "1118g"],
          columns=["445g", "692g", "1118g"],
      )

      plt.figure(dpi=700)
      sb.heatmap(df_mtrix, annot=True, fmt="g", square=True, cmap="Blues")
      plt.title("transformer_classifier")
      plt.xlabel("Predictit label", fontsize=13)
      plt.ylabel("True label", fontsize=13)
      plt.savefig(f"{home}/result/transformer-classifier/transformer_classifier_matrix.png")


  # %%
  # 各データのカウントができないので変形
  t_test_change = []
  for i in range(60):
      t_test_change.append(np.argmax(t_test[i]))

  # 混合行列に使用するデータを格納
  predict_prob = model.predict(x_test)
  predict_classes = np.argmax(predict_prob, axis=1)
  true_classes = t_test_change

  # 混合行列生成
  print_mtrix(true_classes, predict_classes)

if __name__ == '__main__':
  main()