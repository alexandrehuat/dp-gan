import os.path as osp
import argparse
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D
from .dpgan import BasicDPGAN
from .pretraining import mnist_models, MODELS_PATHS, data


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=1, help="the training epochs (set to 0 if you want to test dp-GAN only")
    parser.add_argument("-l", "--lam", type=float, default=10, help="the improved WGAN regularization coefficient")
    parser.add_argument("-t", "--critic", type=int, default=4, help="the number of discriminator iterations per generator iteration")
    parser.add_argument("-m", "--batch", type=int, default=64, help="the batch size")
    parser.add_argument("-A", "--adam", type=float, nargs=3, default=(0.002, 0.5, 0.9), help="the adam optimizer parameters (alpha, beta1, beta2)")
    parser.add_argument("-C", "--clip", type=float, default=0.9, help="the gradient clipping bound")
    parser.add_argument("-s", "--sigma", type=float, default=1, help="the noise scale")
    parser.add_argument("-E", "--eps0", type=float, default=4, help="the epsilon privacy budget")
    parser.add_argument("-D", "--delta0", type=float, default=1e-05, help="the delta privacy budget")
    parser.add_argument("--tol", type=float, default=1e-08, help="the tolerance for training convergence")

    return parser.parse_args()


def _evaluate(G, D, X, n=10):
    """
    Evaluates the nets by plotting some examples.
    """
    # Preparing examples
    im_shape = X.shape[1:]
    X_ = X[rdm.permutation(X.shape[0])[:n]]
    Z = rdm.uniform(size=(n,) + im_shape)
    Gz = G.predict(Z)
    y_pred = np.empty((3, n))
    y_pred[0, :] = D.predict_proba(X_).ravel()
    rho = rdm.uniform(size=n)
    X_hat = np.array([(rho[i]*X_[i]+(1-rho[i])*Z[i]).tolist() for i in range(n)])
    y_pred[1, :] = D.predict_proba(X_hat).ravel()
    y_pred[2, :] = D.predict_proba(Gz).ravel()

    # Plotting
    fig, axs = plt.subplots(3, n, figsize=np.array((n, 3))*1.4, sharex=True, sharey=True)
    for i in range(3):
        for j in range(n):
            ax = axs[i, j]
            im = np.squeeze(X_[j] if i == 0
                            else X_hat[j] if i == 1
                            else Gz[j])
            ax.imshow(im, cmap=plt.cm.Greys)
            if j == 0:
                ax.set_title("D(x)=" if i == 0
                             else "D(\hat x)=" if i == 1
                             else "D(G(z))=")
            ax.set_title("${}{:.2f}$".format(ax.get_title(), y_pred[i, j]))
    fig.tight_layout()
    fig.savefig("summary/evaluate_dpgan.png")
    plt.show()


if __name__ == "__main__":
    args = _parse_args()

    # Loading data
    X_train, X_test, _, _ = data()
    im_shape = X_train.shape[1:]

    # Instanciating the neural nets
    G, D = mnist_models()
    # Adapting the discriminator net
    D.pop()
    D.add(Dense(1, activation="sigmoid", name="is_real"))
    # Freezing the convolutionnal layers
    for i in range(len(D.layers)):
        if isinstance(D.layers[i], Conv2D):
            D.layers[i].trainable = False
    # D.compile("adam", "mse")  # confirm layers modifications (see the Keras tutorial)
    # D.fit(X_train[:1], np.ones((1, 1)))

    try:
        # dp-GAN training
        if args.epochs > 0:
            save_paths = []
            for path, net in zip(MODELS_PATHS, ["G", "D"]):
                dir, base = osp.split(path)
                base = "_".join(("DPGAN", net, base))
                save_paths.append(osp.join(dir, base))
            train_kw = {"epochs": args.epochs,
                        "lam": args.lam,
                        "n_critic": args.critic,
                        "batch_size": args.batch,
                        "optimizer": tf.train.AdamOptimizer(*args.adam),
                        "C": args.clip,
                        "sigma": args.sigma,
                        "eps0": args.eps0,
                        "delta0": args.delta0,
                        "save_paths": save_paths,
                        "tol": args.tol}

            dpgan = BasicDPGAN(G, D)
            G, D = dpgan.train(X_train, **train_kw)
    finally:
        print("Evaluating the nets")
        _evaluate(G, D, X_test)
