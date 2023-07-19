#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb  # 混合行列

from sklearn.model_selection import train_test_split  # データセットの分割
from sklearn.metrics import confusion_matrix  # 混合行列の計算
from keras.models import Sequential
from keras.layers  import Dense
from keras.layers import Bidirectional  # 双方向
from keras.layers import LSTM
from keras.layers import Flatten
from keras_self_attention import SeqSelfAttention
from keras.layers.core import Activation  # 活性化関数
from keras.optimizers import Adam  # 最適化関数
from tensorflow.keras.utils import plot_model  # モデル図
from keras.utils import np_utils

# ハイパーパラメータの調整用
from sklearn.model_selection import GridSearchCV  # これは使う
from keras.wrappers.scikit_learn import KerasRegressor  # これを使うのは間違っていたかもしれない

#%%
# csvファイル読み込み
# BOM付きなのでencoding="utf_8_sig"を指定
csv100 = np.loadtxt("csv/100.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
csv200 = np.loadtxt("csv/200.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
csv300 = np.loadtxt("csv/300.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
csv500 = np.loadtxt("csv/500.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
csv600 = np.loadtxt("csv/600.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
csv700 = np.loadtxt("csv/700.csv", delimiter=",", encoding="utf_8_sig", unpack=True)

# 時間の行を削除
csv100 = np.delete(csv100, 0, 0)
csv200 = np.delete(csv200, 0, 0)
csv300 = np.delete(csv300, 0, 0)
csv500 = np.delete(csv500, 0, 0)
csv600 = np.delete(csv600, 0, 0)
csv700 = np.delete(csv700, 0, 0)
# %%
# データを格納、学習に使う長さを指定
length = 101

data = []  # 入力値
target = []  # 教師データ

# 入力値と教師データを格納
for i in range(csv100.shape[0]):  # データの数
    data.append(csv100[i][0:length])
    target.append(0)
for i in range(csv200.shape[0]):
    data.append(csv200[i][0:length])
    target.append(1)
for i in range(csv300.shape[0]):
    data.append(csv300[i][0:length])
    target.append(2)
for i in range(csv500.shape[0]):
    data.append(csv500[i][0:length])
    target.append(3)
for i in range(csv600.shape[0]):
    data.append(csv600[i][0:length])
    target.append(4)
for i in range(csv700.shape[0]):
    data.append(csv700[i][0:length])
    target.append(5)
# %%
# kerasで学習できる形に変換
# リストから配列に変換
x = np.array(data).reshape(len(data), length, 1)
t = np.array(target).reshape(len(target), 1)
t = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

# 訓練データ、検証データ、テストデータに分割
x_train, x_test, t_train, t_test = train_test_split(
    x, t, test_size=int(len(data) * 0.4), stratify=t
)
x_valid, x_test, t_valid, t_test = train_test_split(
    x_test, t_test, test_size=int(len(x_test) * 0.5), stratify=t_test
)
# %%
# 入力、隠れ、出力のノード数
l_in = len(x[0])  # 101
l_hidden = 30
l_out = 6
#%%
"""
#ハイパーパラメータ調整
#ハイパーパラメータ調整時は、プログラム実行はここまで
#下の行で構築しているモデルをここで構築する

#モデルを定義
def create_model(lr,l_hidden): #引数は調整したいパラメータ
  model = Sequential([
    Bidirectional(LSTM(l_hidden, input_shape=(l_in, 1),return_sequences=True)),
    SeqSelfAttention(attention_width=l_hidden),
    Flatten(),
    Dense(6,activation='softmax')
  ])
  #最適化関数と評価関数
  optimizer = Adam(lr=lr,beta_1=0.9,beta_2=0.999)
  model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  return model

#調整したいパラメータとその数値
#これはバッチサイズ、隠れ層の数、学習率
batch_size = [16]
l_hidden = [30,50]
lr = [0.01]
param_grid = dict(batch_size=batch_size, l_hidden=l_hidden,lr=lr)

#グリッドサーチのモデルとパラメータを定義
model = KerasRegressor(build_fn=create_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

#一番良いパラメータの組み合わせを総当りで実行
#エポック数：何回目の学習で評価するか:
grid_result = grid.fit(x_valid,t_valid,epochs=1)

#結果を出力、一番良いパラメータの組み合わせが出力される
print(grid_result.best_params_)
"""
# %%
# モデルの構築
# Self-Attentionの時だけsummaryの位置を変えないとエラーが出る
model = Sequential()  # 入力と出力が１つずつ
model.add(
    Bidirectional(LSTM(l_hidden, input_shape=(l_in, 1), return_sequences=True))
)  # 隠れ層のノード数、入力の形、各時間で出力
model.add(SeqSelfAttention(attention_width=15))  # Self-Attentionの隠れ層のノード数
model.add(Flatten())  # 次元を変換
model.add(Dense(l_out))  # 出力層を追加
model.add(Activation("softmax"))  # 多クラス分類なのでソフトマックス関数
#%%
##学習の最適化
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
# 損失関数（交差エントロピー誤差）、最適化関数、評価関数
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

# バッチサイズ、エポック数
batch_size = 32
epochs = 50

# 学習開始
result = model.fit(
    x_train,
    t_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_valid, t_valid),
)

model.summary()  # モデルの詳細を表示
plot_model(model,to_file='result/self-attention/model_self-attention1.png',show_shapes=True) #モデル図

#%%
# 正解率の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["accuracy"], label="train_acc") # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_accuracy"], label="valid_acc") # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("result/self-attention/self-attention_accuracy1.png")
plt.show()
# %%
# 損失関数の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["loss"], label="training_loss") # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_loss"], label="validation_loss") # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("result/self-attention/self-attention_loss1.png")
plt.show()
#%%
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

print(epochs, l_hidden, batch_size)
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
        index=["100g", "200g", "300g", "500g", "600g", "700g"],
        columns=["100g", "200g", "300g", "500g", "600g", "700g"],
    )

    plt.figure(dpi=700)
    sb.heatmap(df_mtrix, annot=True, fmt="g", square=True, cmap="Blues")
    plt.title("LSTM")
    plt.xlabel("Predictit label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.savefig("result/self-attention/self-attention.png")
    plt.show()


#%%
# 各データのカウントができないので変形
t_test_change = []
for i in range(96):
    t_test_change.append(np.argmax(t_test[i]))

# 混合行列に使用するデータを格納
predict_prob = model.predict(x_test)
predict_classes = np.argmax(predict_prob, axis=1)
true_classes = t_test_change

# 混合行列生成
print_mtrix(true_classes, predict_classes)
