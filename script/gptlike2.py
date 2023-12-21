import sys
import os

home = os.environ["HOME"]
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from my_model.mingpt.model import GPT, GPTConfig, TransformerClassifier
from my_model.mingpt.optimization import AdamWeightDecay
import matplotlib.pyplot as plt
import seaborn as sb  # 混合行列
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras import Model, Input, layers
from keras.models import load_model


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

  data = []  # 入力値
  target = []  # 教師データ


  # 入力値と教師データを格納
  for x in range(0, 10):
      for i in range(csv_convex.shape[0]):  # データの数
          tmp = csv_convex[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])  # データ数を半分にしながら挿入
          target.append(0)
      for i in range(csv_cylinder.shape[0]):
          tmp = csv_cylinder[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])
          target.append(1)
      for i in range(csv_wall.shape[0]):
          tmp = csv_wall[i][length_start:length_end] + (0.1 * x)
          data.append(tmp[::skip_num])
          target.append(2)


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

  m_conf = GPTConfig(int((length_end - length_start) / skip_num), int((length_end - length_start) / skip_num), n_layer=12, n_head=8, n_embd=64)

  learning_rate = 3e-4
  betas = (0.9, 0.95)
  grad_norm_clip = 1.0
  weight_decay = 0.1  # only applied on matmul weights
  optimizer = AdamWeightDecay(learning_rate=learning_rate, # type: ignore
                                              weight_decay_rate=weight_decay,
                                              beta_1=betas[0], beta_2=betas[1],
                                              gradient_clip_norm=grad_norm_clip,
                                              exclude_from_weight_decay=['layer_normalization', 'bias'])
  epochs = 1
  batch_size = 1

  model_convex = load_model(f"{home}/model_convex", custom_objects={"AdamWeightDecay": optimizer })
  model_cylinder = load_model(f"{home}/model_cylinder", custom_objects={"AdamWeightDecay": optimizer })
  model_wall = load_model(f"{home}/model_wall", custom_objects={"AdamWeightDecay": optimizer })

  model_convex.trainable = False
  model_cylinder.trainable = False
  model_wall.trainable = False

  inputs = layers.Input((int((length_end - length_start) / skip_num),1,))
  linear = layers.Dense(1, activation="linear")
  concatenated = layers.concatenate([linear(model_convex(inputs)), linear(model_cylinder(inputs)), linear(model_wall(inputs))])
  flatten = layers.Flatten()(concatenated)
  softmaxed = layers.Dense(3, activation="softmax")(flatten)
  model = Model(inputs=inputs, outputs=softmaxed)
  model.summary()
  model.compile(
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )
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
  plt.savefig(f"{home}/result/gptlike/gptlike_accuracy.png")
  # %%
  # 損失関数の可視化
  plt.figure(dpi=700)
  plt.plot(range(1, epochs + 1), result.history["loss"], label="training_loss")  # type: ignore
  plt.plot(range(1, epochs + 1), result.history["val_loss"], label="validation_loss")  # type: ignore
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.savefig(f"{home}/result/gptlike/gptlike_loss.png")
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
      plt.savefig(f"{home}/result/gptlike/gptlike_matrix.png")


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
