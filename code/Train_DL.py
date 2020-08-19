import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

batch_size = 5
epoch = 50


def LoadData():
    # X = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/X.txt", dtype=np.float)
    # y = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/y.txt", dtype=np.int)

    X = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/心肌/X.txt", dtype=np.float)
    y = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/心肌/y.txt", dtype=np.int)

    return X, y


def Preprocessing(X, y):
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train_valid[:-X_test.shape[0]]
    X_valid = X_train_valid[-X_test.shape[0]:]
    y_train = y_train_valid[:-y_test.shape[0]]
    y_valid = y_train_valid[-y_test.shape[0]:]
    print(X_train.shape, X_valid.shape, X_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, X.shape[1], 1)
    X_valid = X_valid.reshape(-1, X.shape[1], 1)
    X_test = X_test.reshape(-1, X.shape[1], 1)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def MakeDataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.repeat(epoch)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)

    return dataset


def Metrics(model, X_test, y_test):
    y_predict_probability = model.predict(X_test)
    y_predict = np.argmax(y_predict_probability, axis=1)

    precision = precision_score(y_test, y_predict, average="weighted")
    recall = recall_score(y_test, y_predict, average="weighted")
    f1 = f1_score(y_test, y_predict, average="weighted")
    print("precision =", precision)
    print("recall =", recall)
    print("f1 =", f1)


if __name__ == '__main__':
    X, y = LoadData()

    X_train, X_valid, X_test, y_train, y_valid, y_test = Preprocessing(X, y)

    dataset_train = MakeDataset(X_train, y_train)
    dataset_valid = MakeDataset(X_valid, y_valid)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, padding="valid", activation="selu",
                                     input_shape=[X.shape[1], 1]))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv1D(64, kernel_size=2, strides=1, padding="valid", activation="selu"))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="selu"))
    model.add(tf.keras.layers.Dense(64, activation="selu"))
    model.add(tf.keras.layers.Dense(32, activation="selu"))
    model.add(tf.keras.layers.Dense(16, activation="selu"))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    history = model.fit(dataset_train, epochs=epoch, steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=dataset_valid, validation_steps=X_valid.shape[0] // batch_size)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    Metrics(model, X_test, y_test)
