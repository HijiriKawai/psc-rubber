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
from keras.utils import plot_model  # モデル図
from keras.utils import np_utils

# ハイパーパラメータの調整用
from sklearn.model_selection import GridSearchCV  # これは使う
from keras.wrappers.scikit_learn import KerasRegressor  # これを使うのは間違っていたかもしれない

#%%
# csvファイル読み込み
# BOM付きなのでencoding="utf_8_sig"を指定

#445g
csv_convex = np.loadtxt("./data/convex.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
#692g
csv_cylinder = np.loadtxt("./data/cylinder.csv", delimiter=",", encoding="utf_8_sig", unpack=True)
#1118g
csv_wall = np.loadtxt("./data/wall.csv", delimiter=",", encoding="utf_8_sig", unpack=True)



# 時間の行を削除
csv_convex = np.delete(csv_convex, 0, 0)
csv_cylinder = np.delete(csv_cylinder, 0, 0)
csv_wall = np.delete(csv_wall, 0, 0)

# %%
# データを格納、学習に使う長さを指定
length_start = 1200
length_end = 3000

data = []  # 入力値
target = []  # 教師データ

# 入力値と教師データを格納
for i in range(csv_convex.shape[0]):  # データの数
    data.append(csv_convex[i][length_start:length_end])
    target.append(0)
for i in range(csv_cylinder.shape[0]):
    data.append(csv_cylinder[i][length_start:length_end])
    target.append(1)
for i in range(csv_wall.shape[0]):
    data.append(csv_wall[i][length_start:length_end])
    target.append(2)

# %%
# kerasで学習できる形に変換
# リストから配列に変換
x = np.array(data).reshape(len(data), length_end - length_start, 1)
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
l_in = len(x[0])
l_hidden = 30
l_out = 3
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
plot_model(model,to_file='result/self-attention/self-attention_model.png',show_shapes=True) #モデル図

#%%
# 正解率の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["accuracy"], label="train_acc") # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_accuracy"], label="valid_acc") # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("result/self-attention/self-attention_accuracy.png")
plt.show()
# %%
# 損失関数の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["loss"], label="training_loss") # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_loss"], label="validation_loss") # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("result/self-attention/self-attention_loss.png")
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
        index=["445g", "692g", "1118g"],
        columns=["445g", "692g", "1118g"],
    )

    plt.figure(dpi=700)
    sb.heatmap(df_mtrix, annot=True, fmt="g", square=True, cmap="Blues")
    plt.title("LSTM")
    plt.xlabel("Predictit label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.savefig("result/self-attention/self-attention_matrix.png")
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
