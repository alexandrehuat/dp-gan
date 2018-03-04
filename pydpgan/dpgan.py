import abc
import numpy as np
import numpy.random as rdm
import tensorflow as tf
from tensorflow.train import AdamOptimizer

TF_FLOAT = tf.float64


class DPGAN(abc.ABC):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator


class BasicDPGAN(DPGAN):
    def __init__(self, generator, discriminator):
        super().__init__(generator, discriminator)


    def train(self, X, Z=np.random.uniform, epochs=100, lam=0.01, n_critic=100, batch_size=100, optimizer=AdamOptimizer(), C=1, sigma=1, eps0=10, delta0=1e-05):
        # Building the training tensor flow
        x = tf.placeholder(TF_FLOAT, self.discriminator.input_shape)
        rho = tf.random_uniform((1,), 0, 1, TF_FLOAT)
        z = tf.placeholder(TF_FLOAT, self.generator.input_shape)
        x_hat = tf.multiply(rho, x) + tf.multiply(tf.constant(1) - rho, self.generator(z))
        loss_i = self.discriminator(self.generator(z)) - self.discriminator(x) + tf.multiply(tf.constant(lam), tf.square(tf.norm(tf.gradients(self.discriminator(x_hat), x_hat)) - tf.constant(1)))  # differentiable by w
        grad_i = tf.gradients(loss_i, self.discriminator.weights)
        ksi = tf.random_normal(grad_i.shape, 0, tf.multiply((sigma * C) ** 2, tf.eye(*grad_i.shape)))
        grad_i = grad_i / tf.maximum(tf.constant(1), tf.norm(grad_i) / tf.constant(C)) + tf.constant(ksi)

        # Initializing the local variables
        e = 0
        delta = 0

        # Training loop
        with tf.Session() as sess:
            while e < epochs and delta <= delta0:  # and the generator has not converged
                for t in range(n_critic):
                    batch_grads = []
                    for i in range(batch_size):
                        batch_grads.append(sess.run(grad_i, {x: X[rdm.choice(X.shape[0]), :], z: Z()}))

                    # Updating the privacy accountant
                    # ...

                    # Updating the discriminator
                    # ...

                e += 1