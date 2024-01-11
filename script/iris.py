from unittest import skip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras import layers, Model, Input
from keras.utils import to_categorical, np_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
import os

home = os.environ["HOME"]


def print_mtrix(t_true, t_predict, length_start, length_end, iter_num):
    mtrix_data = confusion_matrix(t_true, t_predict)
    df_mtrix = pd.DataFrame(
        mtrix_data,
        index=["445g", "692g", "1118g"],
        columns=["445g", "692g", "1118g"],
    )

    plt.figure(dpi=700)
    sb.heatmap(df_mtrix, annot=True, fmt="g", square=True, cmap="Blues")
    plt.title("TRANSFORMERBLOCK")
    plt.xlabel("Predictit label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.savefig(f"{home}/result/iris/iris_{length_start}_to_{length_end}_matrix_{iter_num:02}.png")


# Transformer用のMultiHeadAttentionレイヤーを定義
class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0

        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.embed_dim // self.num_heads)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


# Transformerブロックを定義
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # rate:ドロップアウトの割合
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# モデルの構築
class TransformerClassifier(Model):
    def __init__(
        self,
        num_classes,
        embed_dim,
        num_heads,
        ff_dim,
        input_shape,
        num_layers=2,
        rate=0.1,
    ):
        super(TransformerClassifier, self).__init__()
        self.embedding = layers.Dense(embed_dim, input_shape=input_shape)
        # self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(rate)
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        # x = self.transformer_block(x, training)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)
    def my_summary(self, input_shape, training) :
       tmp_x = Input(shape=input_shape, name='input')
       tmp_m = Model(inputs=[tmp_x], outputs=self.call(tmp_x, training), name='transformer_block')
       tmp_m.summary()
       del tmp_x, tmp_m


# カスタムコールバックの定義
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            model_name = f"transformer_iris_epoch_{epoch + 1}"
            self.model.save(model_name, save_format="tf")  # save_format="tf" を追加
            print(f"\nModel saved as {model_name}\n")


def task(length_start = 1500, length_end = 3000, iter_num = 0):
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
    skip_num = 2

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

    # %%
    # kerasで学習できる形に変換
    # リストから配列に変換
    x = np.array(data).reshape(len(data), int((length_end - length_start) / skip_num))
    print(x)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    t = np.array(target).reshape(len(target), 1)
    t = np_utils.to_categorical(t)  # 教師データをone-hot表現に変換

    # 訓練データ、検証データ、テストデータに分割
    x_train, x_test, t_train, t_test = train_test_split(
        x, t, test_size=int(len(data) * 0.4), stratify=t
    )
    x_valid, x_test, t_valid, t_test = train_test_split(
        x_test, t_test, test_size=int(len(x_test) * 0.5), stratify=t_test
    )

    # データセットの読み込み
    # iris = load_iris()
    # data = iris.data
    # target = to_categorical(iris.target)

    # データの前処理
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    # データを訓練用とテスト用に分割
    # train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=42)

    # ハイパーパラメータ
    embed_dim = 64  # 埋め込み次元数. num_headsで割り切れる数じゃないとダメ
    num_heads = 8  # マルチヘッドアテンション内のヘッド数
    ff_dim = 64  # フィードフォワードネットワークの中間層のニューロン数
    num_classes = 3  # 分類するクラス数
    num_layers = 6  # Transformerの段数

    # モデルを保存する頻度を指定
    save_freq = 100
    custom_checkpoint = CustomModelCheckpoint(save_freq)

    # モデルのインスタンス化
    model = TransformerClassifier(
        num_classes,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers=num_layers,
        input_shape=(x_train.shape[1],),
    )

    # モデルのコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    """
    # 保存されたモデルをロード
    model_name = "transformer_iris_epoch_100"  # この部分は、適切なモデル名に変更してください
    model = tf.keras.models.load_model(model_name)

    # モデルの要約を表示
    model.summary()

    # モデルのコンパイル
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=["accuracy"])
    """

    # モデルの訓練
    history = model.fit(
        x_train, t_train, batch_size=12, epochs=100, validation_split=0.2
    )

    model.my_summary(x_train.shape, True)
    # モデルの評価
    loss, accuracy = model.evaluate(x_test, t_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    t_test_change = []
    for i in range(len(t_test)):
        t_test_change.append(np.argmax(t_test[i]))

    # 混合行列に使用するデータを格納
    predict_prob = model.predict(x_test)
    predict_classes = np.argmax(predict_prob, axis=1)
    true_classes = t_test_change

    print_mtrix(true_classes, predict_classes, length_start, length_end, iter_num)

def main():
    for i in range(0, 10): 
        task(length_start=0, length_end=40, iter_num=i)

if __name__ == "__main__":
    main()
