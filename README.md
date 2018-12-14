# Deep Learning Project: Differentially Private Releasing via Deep Generative Model
## Alexandre Huat (INSA Rouen Normandie, Dept. Information Systems Architectures, Data Science MSc by Research)

----

This repository contains my implementation of dp-GAN (Differentially Private Generative Adversarial Network) for my deep learning project assessment. It is organized as follows:
* Directory `summary` contains the report of the project. This document consists in (i) a summary of the original paper of dp-GAN and (ii) a report on my implementation. Its PDF version has been precompiled, but run `./compile.sh` if needed.
* Directory `dpgan` contains the implementation of the project. See the report in `summary` or the docstrings of each files to understand it.
* Directory `data` contains all relevant data for the use of the implemented neural networks.

All other useful information can be found in `summary/summary.pdf`.

### Requirements

In a Python 3 virtual environment, run `pip install -r requirements.txt`.

The test of dp-GAN requires the MNIST dataset. Since it needs a prior download from Keras, its first run could be long.

### Reference
X. Zhang, S. Ji and T. Wang, "Differentially Private Releasing via Deep Generative Model", _ArXiv e-prints_, jan. 2018. arXiv : [1801.01594 [cs.CR]](https://arxiv.org/abs/1801.01594).
