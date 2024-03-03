# Non-Linearites in Quantum Neural Networks

## Overview
This repository was created to experiment in creating non-linearities with a quantum neural network (QNN). The current body of literature suggests that a model apt to run on near term quantum devices, Noisy Intermediate-Scale Quantum (NISQ) devices, are a family of hybrid classical quantum models known as variational quantum models. It follows that a relevent analysis of the current prospects of QNNs should look primarily to these types of models; despite the recent advancements in deeveloping a fully-quantum perceptron model using approximations of non-linear activation functions [Maronese, M. 2019](https://rdcu.be/dAbUc). 


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