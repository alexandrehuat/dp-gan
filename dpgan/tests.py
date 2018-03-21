import os.path as osp
import argparse
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import numpy.random as rdm
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D
from keras.models import load_model
from keras import backend as K
from .dpgan import BasicDPGAN
from .pretraining import mnist_models, MODELS_PATHS, data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=1, help="the training epochs (set to 0 if you want to test dp-GAN only")
    parser.add_argument("-l", "--lam", type=float, default=10., help="the improved WGAN regularization coefficient")
    parser.add_argument("-t", "--critic", type=int, default=4, help="the number of discriminator iterations per generator iteration")
    parser.add_argument("-m", "--batch", type=int, default=64, help="the batch size")
    parser.add_argument("-A", "--adam", type=float, nargs=3, default=(0.002, 0.5, 0.9), help="the adam optimizer parameters (alpha, beta1, beta2)")
    parser.add_argument("-C", "--clip", type=float, default=1., help="the gradient clipping bound")
    parser.add_argument("-s", "--sigma", type=float, default=1., help="the noise scale")
    parser.add_argument("-e0", "--eps0", type=float, default=4., help="the epsilon privacy budget")
    parser.add_argument("-d0", "--delta0", type=float, default=1e-05, help="the delta privacy budget")
    parser.add_argument("-G", "--G_path", type=str, default=MODELS_PATHS[0], help="the path of the generator model to load")
    parser.add_argument("-D", "--D_path", type=str, default=MODELS_PATHS[1], help="the path of the discriminator model to load")
    parser.add_argument("--tol", type=float, default=1e-08, help="the tolerance for training convergence")

    return parser.parse_args()


def mean_overestimation(y_true, y_pred):
    return K.mean(y_pred - y_true)


def evaluate(G, D, X, n=10):
    """
    Evaluates the nets.
    """
    # Quantitative evaluation: Perf on real and fake data
    # mape = mean_absolute_percentage_error
    D.compile("adam", mean_overestimation)
    m = D.evaluate(X, np.ones((X.shape[0], 1)))
    pretitle = "Discriminator's mean overestimation of"
    print(pretitle, "X_test: {:.4f}".format(m))
    Z = rdm.uniform(size=X.shape)
    m = D.evaluate(Z, np.zeros((Z.shape[0], 1)))
    print(pretitle, "Z: {:.4f}".format(m))
    Gz = G.predict(Z)
    m = D.evaluate(Gz, np.zeros((Gz.shape[0], 1)))
    print(pretitle, "G(z): {:.4f}".format(m))

    # Qualitative evaluation: Plotting some examples
    im_shape = X.shape[1:]
    X_ = X[rdm.permutation(X.shape[0])[:n]]
    Z, Gz = Z[:n], Gz[:n]
    y_pred = np.empty((3, n))
    y_pred[0, :] = D.predict_proba(X_).ravel()
    rho = rdm.uniform(size=n)
    X_hat = np.array([(rho[i]*X_[i]+(1-rho[i])*Z[i]).tolist() for i in range(n)])
    y_pred[1, :] = D.predict_proba(X_hat).ravel()
    y_pred[2, :] = D.predict_proba(Gz).ravel()

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
    return fig


if __name__ == "__main__":
    args = parse_args()

    # Loading data
    X_train, X_test, _, _ = data()
    im_shape = X_train.shape[1:]

    # Instanciating the neural nets
    G, D = mnist_models(pretrained=False)
    if osp.exists(args.G_path):
        print("Loading model G")
        G = load_model(args.G_path)
    if osp.exists(args.D_path):
        print("Loading model D")
        D = load_model(args.D_path)
    # Adapting the discriminator net
    D.pop()
    D.add(Dense(1, activation="sigmoid", name="is_real"))
    # Freezing the convolutionnal layers
    # for i in range(len(D.layers)):
    #     if isinstance(D.layers[i], Conv2D):
    #         D.layers[i].trainable = False

    try:
        # dp-GAN training
        if args.epochs > 0:
            save_paths = []
            for path, net in zip([args.G_path, args.D_path], ["G", "D"]):
                dir, base = osp.split(path)
                if "DPGAN" not in base:
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
            G, D, G_loss_hist, D_loss_hist = dpgan.train(X_train, **train_kw)
    finally:
        print("Evaluating the nets")
        fig = evaluate(G, D, X_test)
        path = "summary/dpgan_evaluation_t{}_l{}_a{}_b1{}_b2{}_C{}_s{}".format(args.critic, args.lam, *args.adam, args.clip, args.sigma)
        if osp.exists(path + ".png"):
            path += "_bis"
        path += ".png"
        fig.savefig(path)

        # Plotting the training losses
        fig, ax = plt.subplots()
        xx = np.arange(len(D_loss_hist))+1
        ax.plot(xx[::args.critic]+args.critic-1, G_loss_hist, "-o", label="G's loss", ms=3.6)
        ax.plot(xx, D_loss_hist, "-x", label="D's loss", ms=3.6)
        ax.grid(axis="y")
        ax.set_xlabel("Batch iteration")
        ax.set_title(r"Parameters: $n_\mathit{critic}"+r"={}, \lambda={}, C={}, \sigma={},$".format(args.critic, args.lam, args.clip, args.sigma)+"\n"+r"$\alpha={}, \beta_1={}, \beta_2={}$".format(*args.adam))
        ax.legend(loc="upper right")
        path = "summary/dpgan_loss_history_t{}_l{}_a{}_b1{}_b2{}_C{}_s{}".format(args.critic, args.lam, *args.adam, args.clip, args.sigma)
        if osp.exists(path + ".png"):
            path += "_bis"
        path += ".png"
        fig.savefig(path)

        print("Close the figures to quit")
        plt.show()
