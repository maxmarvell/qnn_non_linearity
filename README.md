# Non-Linearites in Quantum Neural Networks

## Overview
This repository was created to experiment in creating non-linearities with a quantum neural network (QNN). The current body of literature suggests that a model apt to run on near term quantum devices, Noisy Intermediate-Scale Quantum (NISQ) devices, are a family of hybrid classical quantum models known as variational quantum models or variational quantum circuits (VQCs). It follows that a relevent analysis of the current prospects of QNNs should look primarily to these types of models; despite the recent advancements in developing a fully-quantum perceptron model using approximations of non-linear activation functions [Maronese, M. 2019](https://rdcu.be/dAbUc). 

The contents of this repository follow as:

1. A repository ```non_linear``` containing all source code used to generate data for this project

    1. A class ```models``` containing four different elementary VQC units which vary by measurement procedure, ansatz structure and classical data upload method.

    2. A class ```quanvolutional``` which creates a convolutional neural network from the elementary units. A classcial activation funcition can also be optionally applied using the "ACAS" method.

    3. A class ```classifier``` which takes a VQC model and a classification task and trains the variational circuit to it.

    4. A class ```fourier``` which takes a VQC and samples the fourier coefficents of the model for random parametrisations.

    4. A class ```fisher``` which takes a VQC and samples eigenvalues of the classical fisher information matrix.

    5. A class ```qfisher``` which takes a VQC and samples eigenvalues of the quantum information matrix.

    6. A helper class ```autoencoder``` to autoencoder classification tasks, either reducing or enhancing the size of the feature space with loss of accuracy to the original model.

2. A repository ```graphs``` containing all of the graphs generated and used for this project


## Background

## Usage
How to use the Library
```
from non_linear import models
from non_linear.autoencoder import Autoencoder
from non_linear.classifier import ClassifierQNN
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import make_moons
import torch
```

![Sample image of a classification class using the data reuploading](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classifier/data_reupload/sample.svg?raw=true)


## Results and Discussion