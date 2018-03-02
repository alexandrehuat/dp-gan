import abc
import numpy as np
import tensorflow as tf


class DPGAN(abc.ABC):
    def __init__(self, G, D):
        self.G = G
        self.D = D
        self.session = tf.Session()


class BasicDPGAN(DPGAN):
    def __init__(self, G, D):
        super().__init__(G, D)

    def train(self, X, Z="uniform", epochs=10, lam=0.01, n_critic=100, m=100, optimizer=tf.train.AdamOptimizer(), C=1, sigma=1, eps0=10, delta0=1e-05):
        # Init
        e = 0

        # Training loop
        while e < epochs:  # or has not converged
            for t in range(n_critic):
                grad = tf.Tensor()
                for i in range(m):
                    # Sampling data
                    x = np.random.choice(data)
                    z = np.random.uniform()

                    x = tf.placeholder(name="x")
                    rho = tf.random_uniform(shape=(1,), name="rho")
                    z = tf.placeholder(name="z")
                    x_hat = tf.multiply(rho, x) + tf.multiply((tf.constant(1) - rho), self.G(z))

                    # Updating the discriminator
                    loss_i = self.D(self.G(z)) - self.D(x) + tf.multiply(lam, tf.square(tf.norm(tf.gradients(self.D(x_hat), var=[x_hat])) - tf.constant(1)))  # must be derivable by w
                    grad_i = tf.gradients(loss_i, self.D.weights)
                    ksi = tf.random_normal(grad_i.shape, 0, tf.multiply((sigma * C) ** 2, tf.eye(*grad_i.shape)))
                    grad_i = grad_i / tf.maximum(tf.constant(1), tf.norm(grad_i) / tf.constant(C)) + tf.constant(ksi)

                    batch_grads = tf.concat(batch_grads, grad_i)



            e += 1