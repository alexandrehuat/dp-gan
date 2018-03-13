import abc
import os.path as osp
import numpy as np
import numpy.random as rdm
import tensorflow as tf
from datetime import datetime as dt

TF_FLOAT = tf.float32


def _tic():
    return dt.now()


def _toc(tic):
    return dt.now() - tic


class DPGAN(abc.ABC):
    def __init__(self, G, D):
        self.G = G
        self.D = D

    @abc.abstractmethod
    def train(self):
        pass

    def load_weights(self, G_path=None, D_path=None):
        if G_path and osp.exists(G_path):
            self.G.load_weights(G_path)
        if D_path and osp.exists(D_path):
            self.D.load_weights(D_path)

    def save_weights(self, G_path=None, D_path=None):
        if G_path:
            self.G.save_weights(G_path)
        if D_path:
            self.D.save_weights(D_path)


class BasicDPGAN(DPGAN):
    def __init__(self, G, D):
        super().__init__(G, D)

    def train(self, X, epochs=100, lam=10, n_critic=4, batch_size=64, optimizer=tf.train.AdamOptimizer(0.002, 0.5, 0.9), C=1, sigma=1, eps0=4, delta0=1e-05, tol=1e-08, save_paths=None, verbose=1):
        # Building the training tensor flow
        if verbose > 0:
            print("Building the tensor flow")
        x = tf.placeholder(TF_FLOAT, self.D.input_shape)
        rho = tf.random_uniform((batch_size,), dtype=TF_FLOAT)
        for d in x.shape[1:].as_list(): rho = tf.stack(d * [rho], axis=-1)  # reshaping rho to z's shape
        z = tf.random_uniform((batch_size,) + self.G.input_shape[1:], dtype=TF_FLOAT, name="z")
        x_hat = tf.multiply(rho, x) + tf.multiply(tf.constant(1.) - rho, self.G(z))
        D_grad_x_hat = tf.gradients(self.D(x_hat), x_hat)[0]
        D_loss = self.D(self.G(z)) - self.D(x) + tf.multiply(tf.constant(lam, TF_FLOAT), tf.square(tf.norm(D_grad_x_hat, axis=1 if x_hat._rank() <= 2 else (1, 2)) - tf.constant(1.)))
        D_grad_vars = optimizer.compute_gradients(D_loss, self.D.trainable_weights)
        # import pdb; pdb.set_trace()
        D_grad_vars = [(tf.divide(grad, tf.maximum(tf.constant(1.), tf.divide(tf.norm(grad), tf.constant(C, TF_FLOAT)))) + tf.random_normal(grad.shape, 0, (sigma * C) ** 2), var) for grad, var in D_grad_vars] # Clipping and noising
        # D_grad_vars = [(tf.reduce_mean(grad, axis=0), var) for grad, var in D_grad_vars]
        train_D = optimizer.apply_gradients(D_grad_vars)

        G_loss = tf.reduce_mean(-self.D(self.G(z)), axis=0)
        train_G = optimizer.minimize(G_loss, var_list=self.G.trainable_weights)

        # Init
        if verbose > 0:
            print("Initializing variables")
        delta = 0
        init_all = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_all)

        # Training loop
            if verbose > 0:
                print("Training")
            tic = _tic()
            for e in range(1, epochs+1):
                if verbose > 0:
                    print("epoch={}/{} (time={})".format(e, epochs, _toc(tic)), end="\r")
                for t in range(1, n_critic+1):
                    if verbose > 0:
                        print("epoch={}/{}, training D (critic_iter={}/{}) (time={})".\
                          format(e, epochs, t, n_critic, _toc(tic)), end="\r")
                    # batch_grads = []
                    # for i in range(batch_size):
                    #     batch_grads.append(sess.run(D_grad, {x: X[rdm.choice(X.shape[0])]}))

                    # Updating the privacy accountant
                    # with sigma, batch_size and self.D.count_params()

                    # Updating the D

                    sess.run(train_D, {x: X[rdm.choice(X.shape[0], batch_size)]})

                # Updating the G
                if verbose > 0:
                    print("epoch={}/{}, training G (time={})".\
                      format(e, epochs, _toc(tic)) + 24 * " ", end="\r")
                theta_old = self.G.trainable_weights
                sess.run(train_G)

                # Stopping criterias
                G_convergence = all(sess.run(tf.norm(self.G.trainable_weights[i] - theta_old[i]) <= tol) for i in range(len(self.G.trainable_weights)))

                # Saving the models
                if save_paths is not None:
                    if verbose > 0:
                        print("/!\ SAVING THE NEURAL NETS /!\ ".format(e, epochs) + 32 * " ", end="\r")
                    self.save_weights(*save_paths)

                # delta = request A with eps0

                if delta > delta0 or G_convergence:
                    break

        print("\nDone! (time={})".format(_toc(tic)))
