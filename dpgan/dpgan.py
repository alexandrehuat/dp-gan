import abc
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
        self.tf_session = tf.Session()

    @abc.abstractmethod
    def train(self):
        pass

    def save(self, paths):
        self.G.save(paths[0])
        self.D.save(paths[1])


class BasicDPGAN(DPGAN):
    def __init__(self, G, D):
        super().__init__(G, D)

    def train(self, X, epochs=100, lam=10, n_critic=4, batch_size=64, optimizer=tf.train.AdamOptimizer(0.002, 0.5, 0.9), C=1, sigma=1, eps0=4, delta0=1e-05, tol=1e-08, save_paths=None):
        tic = _tic()
        # Building the training tensor flow
        print("Building the tensor flow")
        x = tf.placeholder(TF_FLOAT, self.D.input_shape)
        rho = tf.random_uniform((batch_size,), dtype=TF_FLOAT)
        for d in x.shape[1:].as_list(): rho = tf.stack(d * [rho], axis=-1)  # reshaping rho to z's shape
        z = tf.random_uniform((batch_size,) + self.G.input_shape[1:], dtype=TF_FLOAT, name="z")
        x_hat = tf.multiply(rho, x) + tf.multiply(tf.constant(1.) - rho, self.G(z))
        D_grad_x_hat = tf.gradients(self.D(x_hat), x_hat)[0]
        D_loss = tf.reduce_mean(self.D(self.G(z)) - self.D(x) + tf.multiply(tf.constant(lam, TF_FLOAT), tf.square(tf.norm(D_grad_x_hat, axis=1 if x_hat._rank() <= 2 else (1, 2)) - tf.constant(1.))), axis=0)
        D_grad_vars = optimizer.compute_gradients(D_loss, self.D.trainable_weights)
        D_grad_vars = [(tf.divide(grad, tf.maximum(tf.constant(1.), tf.divide(tf.norm(grad), tf.constant(C, TF_FLOAT)))) + tf.random_normal(grad.shape, 0, (sigma * C) ** 2), var) for grad, var in D_grad_vars] # Clipping and noising
        # dimension incompatibility: D_grad_vars = [(tf.reduce_mean(grad, axis=0), var) for grad, var in D_grad_vars]
        train_D = optimizer.apply_gradients(D_grad_vars)

        G_loss = tf.reduce_mean(-self.D(self.G(z)), axis=0)
        train_G = optimizer.minimize(G_loss, var_list=self.G.trainable_weights)

        # Init
        print("Initializing variables")
        delta = 0
        self.tf_session.run(tf.global_variables_initializer())
        self.tf_session.run(tf.local_variables_initializer())
        batch_loss_hist = []
        D_loss_hist = []
        G_loss_hist = []

        # Training loop
        print("Training")
        for e in range(1, epochs+1):
            verb = "Epoch {}/{}".format(e, epochs)
            print(verb, end="\r")
            verb += " - D_loss: ["
            for t in range(1, n_critic+1):
                # Updating D
                [loss], _ = self.tf_session.run([D_loss, train_D], {x: X[rdm.choice(X.shape[0], batch_size)]})
                verb += "{:.2f}{}".format(loss, int(t<n_critic)*", ")
                print(verb, end="\r")
                D_loss_hist.append(loss)
            verb += "]"

                # TODO: Updating the privacy accountant with sigma, batch_size and self.D.count_params()

            # Updating G
            theta_old = self.G.trainable_weights
            [loss], _ = self.tf_session.run([G_loss, train_G])
            G_loss_hist.append(loss)
            verb += " - G_loss: [{:.2f}] - Time elapsed: {}".format(loss, _toc(tic))
            print(verb)

            # Saving the models
            if save_paths:
                self.save(save_paths)

            # TODO: delta = request privacy with eps0

            # Stopping criterias
            if tol >= 0:
                G_convergence = all(self.tf_session.run(tf.norm(self.G.trainable_weights[i] - theta_old[i]) <= tol) for i in range(len(self.G.trainable_weights)))
                if (delta > delta0) or G_convergence:
                    break

        # Returning results
        if tol >= 0:
            if G_convergence:
                print("Training stopped: G has converged")
            if delta > delta0:
                print("Training stopped: Privacy budget exceeded (delta={})".format(delta))
        self.tf_session.close()
        return self.G, self.D, G_loss_hist, D_loss_hist
