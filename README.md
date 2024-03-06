# Non-Linearites in Quantum Neural Networks

## Overview
This repository was created to experiment in creating non-linearities with a quantum neural network (QNN). The current body of literature suggests that a model apt to run on near term quantum devices, Noisy Intermediate-Scale Quantum (NISQ) devices, are a family of hybrid classical quantum models known as variational quantum models or variational quantum circuits (VQCs). It follows that a relevent analysis of the current prospects of QNNs should look primarily to these types of models; despite the recent advancements in developing a fully-quantum perceptron model using approximations of non-linear activation functions [Maronese, M. 2019](https://rdcu.be/dAbUc). 

We started by looking at a variety of different models to implement a variational circuit. This included a mitzvah measurement circuit a? regular non 

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

Prior to running the model on a standardised classification library, metrics were utilised to predict the power of each model. Three metrics were utilised. 

The first follows [Schuld, M. 2020](https://doi.org/10.48550/arXiv.2008.08605), by representing the model a as a fourier series we uncover that the power of the model depends on the coefficents the model has access to control. A model with only a few adjustable fourier coefficents is naturally going to act as a weaker arbitrary function approximator.

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/compare_VQC_models.svg?raw=true)

With Convolutional network simple ansatz no data reupload

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/convolution_simple_ansatz.svg?raw=true)

With Convolutional network simple ansatz and data reupload

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/convolution_data_reupload.svg?raw=true)


Sample Fisher Information eigenvalue distributions

![Eigenvalue distribution of classical fisher information matrix](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classical_fisher/compare_VQC_models.svg?raw=true)

