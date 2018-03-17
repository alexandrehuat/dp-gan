"""
This module defines some pretrained generator and discriminator that can be
better trained in dp-GAN later. Run it to pretrain the models.
"""

import sys
import os.path as osp
from datetime import datetime as dt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.ion()

MODELS_PATHS = ("data/models/MNIST_AE_5Dense.h5", "data/models/MNIST_classifier_3Conv3Dense.h5")


def mnist_models():
    im_shape = (28, 28, 1)

    p = MODELS_PATHS[0]
    if osp.exists(p):
        G = load_model(p)
    else:
        G = Sequential()
        units = np.logspace(*np.log2([10, 784]), 3, base=2).round().astype(int)
        G.add(Flatten(input_shape=im_shape))
        for u in reversed(units[:-1]):
            G.add(Dense(u, activation="selu", kernel_initializer="lecun_normal"))
        for u in units[1:]:
            G.add(Dense(u, activation="selu", kernel_initializer="lecun_normal"))
        G.add(Reshape(im_shape))
        G.compile("adam", "mse")

    p = MODELS_PATHS[1]
    if osp.exists(p):
        D = load_model(p)
    else:
        D = Sequential()
        D.add(Conv2D(64, 3, padding="same", activation="selu", kernel_initializer="lecun_normal", input_shape=im_shape))
        D.add(MaxPool2D(padding="same"))
        D.add(Conv2D(64, 3, padding="same", activation="selu", kernel_initializer="lecun_normal"))
        D.add(MaxPool2D(padding="same"))
        D.add(Conv2D(64, 3, padding="same", activation="selu", kernel_initializer="lecun_normal"))
        D.add(MaxPool2D(padding="same"))
        D.add(Flatten())
        D.add(Dense(512, activation="selu", kernel_initializer="lecun_normal"))
        D.add(Dense(512, activation="selu", kernel_initializer="lecun_normal"))
        D.add(Dense(10, activation="softmax"))
        D.compile("adam", "categorical_crossentropy", ["accuracy"])

    return G, D


def train(model, path, x, y, *args, **kwargs):
    model.fit(x, y, *args, **kwargs)
    model.save(path)


def data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis] / 255
    X_test = X_test[..., np.newaxis] / 255
    y_train = to_categorical(y_train).astype(bool)
    y_test = to_categorical(y_test).astype(bool)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    try:
        tic = dt.now()
        GE, DE = int(sys.argv[1]), int(sys.argv[2])  # training epochs to do

        # Loading models
        G, D = mnist_models()

        # Loading data
        X_train, X_test, y_train, y_test = data()

        b, v = 64, 0.1  # batch_size, validation_split
        ge, de = 0, 0  # epochs counters

        # Training loop
        while ge+de < GE+DE:
            if ge < GE:
                print("Generator: Global epoch {}/{} (Time elapsed: {})".format(ge + 1, GE, dt.now() - tic))
                Z = np.random.uniform(size=X_train.shape) - 0.5
                X_tilde = (X_train + Z).clip(0, 1)
                e = 1 if de < DE else GE-ge
                train(G, MODELS_PATHS[0], X_tilde, X_train, b, e, validation_split=v)
                ge += e

            if de < DE:
                print("Discriminator: Global epoch {}/{} (Time elapsed: {})".format(de + 1, DE, dt.now() - tic))
                e = 1 if ge < GE else DE-de
                train(D, MODELS_PATHS[1], X_train, y_train, b, e, validation_split=v)
                de += e
    finally:
        # Quantitative evaluation
        print(80*"*")
        m = G.evaluate(X_test, X_test, b)
        print("Generator's test MSE: {:.4f}".format(m))
        m = D.evaluate(X_test, y_test, b)[1]
        print("Discriminator's test accuracy: {:.4f}".format(m))

        # Qualitative evaluation: Plotting generated samples
        n = 10
        fig, axs = plt.subplots(3, n, figsize=(n, 3), sharex=True, sharey=True)
        for i in range(n):
            x = X_test[y_test[:, i]]
            x = x[[np.random.choice(x.shape[0])]]
            z = np.random.uniform(size=(1,) + X_train.shape[1:])
            x_hat, z = G.predict(x), G.predict(z)
            axs[0, i].imshow(np.squeeze(x), cmap=plt.cm.Greys)
            axs[1, i].imshow(np.squeeze(x_hat), cmap=plt.cm.Greys)
            axs[2, i].imshow(np.squeeze(z), cmap=plt.cm.Greys)
        axs[0, 0].set_ylabel("$x_\mathit{test}$")
        axs[1, 0].set_ylabel("$G(x_\mathit{test})$")
        axs[2, 0].set_ylabel("$G(z)$")
        fig.tight_layout()
        fig.savefig("summary/pretraining_G_{}epochs.png".format(GE))
        print("Close the figures to quit")
        plt.close("all")