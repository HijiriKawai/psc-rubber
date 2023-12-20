from pyexpat import model
import sys
import os

home = os.environ["HOME"]
sys.path.append(f"{home}")

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from my_model.mingpt.model import GPTClaaifier, GPTConfig
from my_model.mingpt.optimization import AdamWeightDecay
import matplotlib.pyplot as plt
import seaborn as sb  # 混合行列
import pandas as pd
from sklearn.metrics import confusion_matrix

class Dataset:
    def __init__(self, data):
        vocab_size = 100  # 3種類の錘
        length_start = 1500
        length_end = 3000

        self.vocab_size = vocab_size
        self.block_size = length_end - length_start
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            item = self.data[i]
            x = item[:-1]
            y = item[1:]

            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)

            yield x, y

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

data = []  # 入力値
target = []  # 教師データ

# 入力値と教師データを格納
for i in range(csv_convex.shape[0]):  # データの数
    tmp = csv_convex[i][length_start:length_end]
    data.append(tmp[::2])  # データ数を半分にしながら挿入
    target.append(0)
for i in range(csv_cylinder.shape[0]):
    tmp = csv_cylinder[i][length_start:length_end]
    data.append(tmp[::2])
    target.append(1)
for i in range(csv_wall.shape[0]):
    tmp = csv_wall[i][length_start:length_end]
    data.append(tmp[::2])
    target.append(2)


# 訓練データ、テストデータに分割
x = np.array(data).reshape(len(data), int((length_end - length_start) / 2), 1)
t = np.array(target).reshape(len(target), 1)
t = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

# 訓練データ、検証データ、テストデータに分割
x_train, x_test, t_train, t_test = train_test_split(
    x, t, test_size=int(len(data) * 0.4), stratify=t
)
x_valid, x_test, t_valid, t_test = train_test_split(
    x_test, t_test, test_size=int(len(x_test) * 0.5), stratify=t_test
)

m_conf = GPTConfig(len(x[0]), 750, n_layer=2, n_head=4, n_embd=128)
model = GPTClaaifier(m_conf, 3)
model.compile(
    optimizer=AdamWeightDecay(), loss="categorical_crossentropy", metrics=["accuracy"]
)

epochs = 10
batch_size = 512

result = model.fit(
    x_train,
    t_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_valid, t_valid),
)

loss, accuracy = model.evaluate(x_test, t_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# %%
# 正解率の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["accuracy"], label="train_acc")  # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_accuracy"], label="valid_acc")  # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(f"{home}/result/self-attention/self-attention_accuracy.png")
# %%
# 損失関数の可視化
plt.figure(dpi=700)
plt.plot(range(1, epochs + 1), result.history["loss"], label="training_loss")  # type: ignore
plt.plot(range(1, epochs + 1), result.history["val_loss"], label="validation_loss")  # type: ignore
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(f"{home}/result/self-attention/self-attention_loss.png")
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
    plt.title("LSTM")
    plt.xlabel("Predictit label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.savefig(f"{home}/result/self-attention/self-attention_matrix.png")


# %%
# 各データのカウントができないので変形
t_test_change = []
for i in range(6):
    t_test_change.append(np.argmax(t_test[i]))

# 混合行列に使用するデータを格納
predict_prob = model.predict(x_test)
predict_classes = np.argmax(predict_prob, axis=1)
true_classes = t_test_change

# 混合行列生成
print_mtrix(true_classes, predict_classes)


# train_dataset_gen = Dataset(x_train)
# test_dataset_gen = Dataset(x_test)

# train_dataset = tf.data.Dataset.from_generator(train_dataset_gen,(tf.float32,tf.float32))
# test_dataset = tf.data.Dataset.from_generator(test_dataset_gen,(tf.float32,tf.float32))

# m_conf = GPTConfig(train_dataset_gen.vocab_size, train_dataset_gen.block_size,
#                   n_layer=2, n_head=4, n_embd=128)
# t_conf = TrainerConfig(max_epochs=5, batch_size=512, learning_rate=6e-4,
#                       lr_decay=True, warmup_tokens=1024, final_tokens=50*len(train_dataset_gen)*(4),
#                       num_workers=4)

# trainer = Trainer(model=GPT,model_config=m_conf,train_dataset=train_dataset,train_dataset_len=len(train_dataset_gen),test_dataset=test_dataset, test_dataset_len=len(test_dataset_gen), config=t_conf)
# trainer.train()

# print("test_start")
# loader = train_dataset.batch(512)
# for b, (x, y) in enumerate(loader):
#   result = sample(trainer.model,x,1,sample=True)
#   print("result:")
#   print(result)