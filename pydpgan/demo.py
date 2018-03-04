import argparse
from keras.models import Sequential
from keras.models import Dense, Conv2D
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from .dpgan import BasicDPGAN


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=100, help="the training epochs")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", help="the dataset")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="the verbosity level (increases with the number of 'v')")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = _parse_args()

    if args.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Instanciating the neural nets
    generator = Sequential()
    generator =

    discriminator = VGG16()
    dpgan = BasicDPGAN(generator, discriminator)

    dpgan.train(X_train)