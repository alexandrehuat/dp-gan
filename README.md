# Deep Learning Project: [Differentially Private Releasing via Deep Generative Model](https://arxiv.org/abs/1801.01594)
## Alexandre Huat
### INSA Rouen Normandie, Dept. Information Systems Architectures, Data Science MSc by Research

### Preliminaries

This repository contains my implementation of dp-GAN for my deep learning project assessment, a Differantially Private Generative Adversarial Network. It is organized as follows:
* Directory `summary` contains the report of the project. Run `./compile.sh` in the directory to produce its PDF version. This document consists in (i) a summary of the original paper of dp-GAN and (ii) a report on my implementation.
* Directory `dpgan` contains the implementation of the project. See the report in `summary` or the docstrings of each files to understand it.
* Directory `data` contains all relevant data for the use of the implemented neural networks.

#### Requirements

In a Python 3 virtual environment, run `pip install -r requirements.txt`.

The test files of dp-GAN requires the MNIST dataset. Then, its first run could be long as it needs a download from Keras.
