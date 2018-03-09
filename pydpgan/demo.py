import argparse
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.advanced_activations import PReLU
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
import numpy.random as rdm
import matplotlib.pyplot as plt
from .dpgan import BasicDPGAN


def _dataset(name="mnist"):
    if args.dataset == "mnist":
        return mnist.load_data()


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=100, help="the training epochs")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="the dataset")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="the verbosity level (increases with the number of 'v')")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _parse_args()

    (X_train, y_train), (X_test, y_test) = _dataset(args.dataset)


    # Instanciating the neural nets
    generator, discriminator = Sequential(), VGG16()

    generator.add(Conv2D(32, 5, input_shape=(28, 28)))
    generator.add(PReLU())
    generator.add(BatchNormalization())
    generator.add(Dropout(0.2))

    generator.add(Conv2D(32, 5))
    generator.add(PReLU())
    generator.add(BatchNormalization())
    generator.add(Dropout(0.2))

    generator.add(Conv2D(32, 5))
    generator.add(PReLU())
    generator.add(BatchNormalization())
    generator.add(Dropout(0.2))

    generator.add(Conv2D(32, 5))
    generator.add(PReLU())
    generator.add(BatchNormalization())
    generator.add(Dropout(0.2))

    generator.add(Conv2D(32, 5, activation="softmax"))

    # Training dp-GAN
    dpgan = BasicDPGAN(generator, discriminator)
    G, D = dpgan.train(X_train)

    # Evaluating the generator by plotting some examples
    Z = rdm.uniform(size=(9,) + X_train.shape)
    im = G.predict(Z)

    plt.figure()
    for i in range(3):
        for j in range(3):
            plt.subplot(i+1, j+1, 3*i+j)
            plt.imshow(im[i+j])
    plt.tight_layout()
    plt.show()

