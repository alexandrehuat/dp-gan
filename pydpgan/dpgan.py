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


    def train(self, X, Z=np.random.uniform, epochs=100, lam=0.01, n_critic=100, batch_size=100, optimizer=AdamOptimizer(), C=1, sigma=1, eps0=10, delta0=1e-05, tol=1e-6):
        # Building the training tensor flow
        x = tf.placeholder(TF_FLOAT, (batch_size,) + self.discriminator.input_shape)
        rho = tf.random_uniform((batch_size,), 0, 1, TF_FLOAT)
        z = tf.placeholder(TF_FLOAT, (batch_size,) + self.generator.input_shape)
        x_hat = tf.multiply(rho, x) + tf.multiply(tf.constant(1) - rho, self.generator(z))
        D_loss = self.discriminator(self.generator(z)) - self.discriminator(x) + tf.multiply(tf.constant(lam), tf.square(tf.norm(tf.gradients(self.discriminator(x_hat), x_hat)) - tf.constant(1)))  # differentiable by w
        D_grad = tf.gradients(D_loss, self.discriminator.weights)
        ksi = tf.random_normal(D_grad.shape, 0, (sigma * C) ** 2)
        grad = tf.divide(D_grad, tf.maximum(tf.constant(1), tf.divide(tf.norm(D_grad), tf.constant(C)))) + ksi
        train_D = optimizer.apply_gradients((tf.reduce_mean(grad, axis=0), self.discriminator.weights))

        G_loss = tf.reduce_mean(-self.discriminator(self.generator(z)), axis=0)
        train_G = optimizer.minimize((G_loss, self.generator.weights))

        # Initializing the local variables
        delta = 0
        init_all = tf.initialize_all_variables()

        # Training loop
        with tf.Session() as sess:
            sess.run(init_all)
            for e in range(epochs):
                for t in range(n_critic):
                    # Updating the privacy accountant
                    # ...

                    # Updating the discriminator
                    sess.run(train_D, {x: X[rdm.choice(X.shape[0], batch_size), :], z: Z(size=batch_size)})

                # Updating the generator
                theta_old = self.generator.weights
                sess.run(train_G, {z: Z(size=batch_size)})

                # Stopping criterias
                G_convergence = sess.run(tf.norm(self.generator.weights - theta_old) <= tol)
                if delta <= delta0 or G_convergence:
                    break