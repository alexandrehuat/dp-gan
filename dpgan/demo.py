import argparse
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Reshape
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt
from .dpgan import BasicDPGAN


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=1000, help="the training epochs")
    parser.add_argument("-l", "--lam", type=float, default=10, help="the improved WGAN regularization coefficient")
    parser.add_argument("-t", "--critic", type=int, default=4, help="the number of discriminator iterations per generator iteration")
    parser.add_argument("-m", "--batch", type=int, default=64, help="the batch size")
    parser.add_argument("-A", "--adam", type=float, nargs=3, default=(0.002, 0.5, 0.9), help="the adam optimizer parameters (alpha, beta1, beta2)")
    parser.add_argument("-C", "--clip", type=float, default=0.9, help="the gradient clipping bound")
    parser.add_argument("-s", "--sigma", type=float, default=1, help="the noise scale")
    parser.add_argument("-E", "--eps0", type=float, default=4, help="the epsilon privacy budget")
    parser.add_argument("-D", "--delta0", type=float, default=1e-05, help="the delta privacy budget")
    parser.add_argument("--tol", type=float, default=1e-08, help="the tolerance for training convergence")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="the dataset")
    parser.add_argument("-T", "--evaluate_only", action="store_true", help="don't train but evaluates the nets only")
    # parser.add_argument("--G_weights", type=str, help="the weights path of the generator")
    # parser.add_argument("--D_weights", type=str, help="the weights path of the discriminator")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="the verbosity level (increases with the number of 'v')")

    args = parser.parse_args()

    return args


def _data(name="mnist"):
    if args.dataset == "mnist":
        (X_train, _), (X_test, _) = mnist.load_data()
        X_train = X_train / 255
        X_test = X_test / 255
        return X_train[..., np.newaxis], X_test[..., np.newaxis]


def _evaluate(G, D, X, n=4):
    # Evaluating the nets by plotting some examples
    im_shape = X.shape[1:]
    n = 4
    Z = rdm.uniform(size=(n,) + im_shape)
    Z = G.predict(Z)
    y_pred = np.ravel(D.predict(np.concatenate([X[rdm.choice(X.shape[0], n)], Z])))

    fig, axs = plt.subplots(2, n, figsize=(8, 4), sharex=True, sharey=True)
    for i in range(2):
        for j in range(n):
            ax = axs[i, j]
            im = np.squeeze(X[j] if i == 0 else Z[j])
            ax.imshow(im, cmap=plt.cm.Greys)
            ax.set_title(r"$D({}) = {:.4f}$".format("x" if i == 0 else "G(z)", y_pred[2*i+j]))
            # ax.set_axis_off()
    axs[0, 0].set_ylabel("real images")
    axs[1, 0].set_ylabel("fake images")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = _parse_args()

    # Loading data
    X_train, X_test = _data(args.dataset)
    im_shape = X_train.shape[1:]

    # Instanciating the neural nets
    G, D = Sequential(), Sequential()

    flat_size = np.prod(im_shape)
    G.add(Flatten(input_shape=im_shape))
    G.add(Dense(10, activation="softmax"))
    for u in np.linspace(10, 784, 10, dtype=int)[1:]:
        G.add(Dense(u, activation="selu"))
    G.add(Reshape(im_shape))

    D.add(Conv2D(64, 5, activation="selu", input_shape=im_shape))
    D.add(MaxPool2D(padding="same"))
    D.add(Conv2D(64, 4, activation="selu"))
    D.add(MaxPool2D(padding="same"))
    D.add(Conv2D(64, 3, activation="selu"))
    D.add(MaxPool2D(padding="same"))
    D.add(Flatten())
    D.add(Dense(10, activation="selu"))
    D.add(Dense(784, activation="selu"))
    D.add(Dense(1, activation="softmax"))
    weights_paths = ("data/weights/G_MNIST.h5", "data/weights/D_3Conv2Dense.h5")  # (args.Gnet, args.Dnet)

    # dp-GAN traning
    dpgan = BasicDPGAN(G, D)
    dpgan.load_weights(*weights_paths)
    try:
        if not args.evaluate_only:
            train_kw = {"epochs": args.epochs,
                        "lam": args.lam,
                        "n_critic": args.critic,
                        "batch_size": args.batch,
                        "optimizer": tf.train.AdamOptimizer(*args.adam),
                        "C": args.clip,
                        "sigma": args.sigma,
                        "eps0": args.eps0,
                        "delta0": args.delta0,
                        "save_paths": weights_paths,
                        "tol": args.tol,
                        "verbose": args.verbose}
            dpgan.train(X_train, **train_kw)
    except (KeyboardInterrupt, SystemExit):
        dpgan.load_weights(*weights_paths)
    finally:
        _evaluate(G, D, X_test, 4)
