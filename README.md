# Non-Linearites in Quantum Neural Networks

## Overview
This repository was created to experiment in creating non-linearities with a quantum neural network (QNN). The current body of literature suggests that a model apt to run on near term quantum devices, Noisy Intermediate-Scale Quantum (NISQ) devices, are a family of hybrid classical quantum models known as variational quantum models or variational quantum circuits (VQCs). It follows that a relevent analysis of the current prospects of QNNs should look primarily to these types of models; despite the recent advancements in developing a fully-quantum perceptron model using approximations of non-linear activation functions [Maronese, M. 2019](https://rdcu.be/dAbUc). 

The end goal of the project is to find a variational quantum model that injects sufficent non-linearities into the system such that the quantum nural network can be assumed a 'universal function approximator'. We started by implementing a variety of different variational circuit models. The purpose of which was to evaluate which model introduced the most non-linear dynamics. In a hybrid classical-quantum circuit the non-linearites are introduced to the system in the measurement phase of the operation. The reasoning behind this is simple, quantum mechanics is fundamentally a composition of linear maps whereas the measurement phase is an inherently non-linear transformation as it is a collapse of superimposed states. 

The standard practice for measurement based quantum computing dictates that measurement are delayed until the end of the computation. As we move to more modern NISQ devices the possibility of mid-circuit measurements has been explored and has already had great theoretical success in quantum error correction. So one thing that was important to investigate is whether mid-circuit measurement would prove appropriate. Other things necessary to verify was whether a uniform repeated ansatz structure made a difference, finally whether it matters if the data is reuploaded to the model. 

Finally the elementary models were fed into a convolutional model where classical non-linear activation functions can be applied to evaluate whether each model benifitted from non-linearites **known** to inject non-linearities. This is one of the proxies we will be using to test whether the model is sufficiently non-linear.

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

## Results and Discussion

Lets begin by running a simple variational circuit with no data-reuploading and simple ansatz structure for a single sample. Here we will suggest that the model has a feature space of size 4, a target space of size 2 (binary classification), and the ansatz repeats 5 times.

```
from non_linear.models import qnn_compiler, simple_ansatz
import jax

N_FEATURES = 4
N_TARGETS = 2
N_LAYERS = 5

# init the compiler
compiler = qnn_compiler(simple_ansatz, N_FEATURES, N_LAYERS, N_TARGETS)

# this model is for classification
qnn = compiler.classification()

input = [.1,.2,.3,.4]
params = jax.random.uniform(jax.random.PRNGKey(0), shape=compiler.parameter_shape)

# this will yield an array of shape (2,) which can be used to classify the input
output = qnn(input, params)
```

Here is the result:

```
output = [Array(0.44702205, dtype=float32), Array(0.09808806, dtype=float32)]
```

Of course these results are completely arbitrary and mean nothing, just that the quantum model produced an output. With this groundwork, however, we can now begin to train the model parameters and see if the model can actually handle a standardised non-linear classifcation task, here we use the make_moons library. As standard we will enhance the feature space of the make_moons library to 4 features, by using an autoencoder, then decode the results where necessary. Here is an example of training a VQC with a simple ansatz structure:

```
from non_linear.models import simple_ansatz
from non_linear.autoencoder import Autoencoder
from non_linear.classifier import ClassifierQNN
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import jax

NOISE = .2
EPOCHS = 100
N_SAMPLES = 500
N_FEATURES = 4
N_LAYERS = 5

# get and encode the make moons data set
data, target = make_moons(n_samples=N_SAMPLES,noise=NOISE)

# encode data
scalerOHE = OneHotEncoder(sparse_output=False)
encoded_target = scalerOHE.fit_transform(target.reshape(-1,1))
  
# encode target
sScaler = StandardScaler()
encoded_data = sScaler.fit_transform(data)

# enhance feature space
autoencoder = Autoencoder(encoded_data, N_FEATURES)
enhanced_data = autoencoder.encoder(torch.from_numpy(encoded_data).float()).detach().numpy()

# train the model
simple_ansatz = ClassifierQNN(models.simple_ansatz, enhanced_data, encoded_target, N_LAYERS)
simple_ansatz.train_test_split()

# score the model
simple_ansatz.learn_model(epochs=100)
simple_ansatz.score_model()
fig = simple_ansatz.plot_fit(show=True, decoded_data=encoded_data)

```

The resulting accuracy and cross entropy loss and accuracy is...

```
Cross entropy loss on training set: 0.5583205819129944
Accuracy of fullmodel on training set: 0.79

Cross entropy loss on test set: 0.5652841925621033
Accuracy of fullmodel on test set: 0.8
```

Hmm a bit dissapointing... how does the classifier look graphically

![classification using simple ansatz](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classifier/simple_ansatz/features=4&layers=5&epochs=100&noise=0.2.svg?raw=true)

It appears that the model is not really learning the classification problem. Rather it is just drawing arbitrary **linear** decision boundaries. Under the representation of quantum model as a partial fourier series we can draw two conclusions. Firstly, that the model is ill-equipped to train the fourier coefficents of the model and secondly that the circuit may have "access" to sufficent fourier coefficents in the first place.

We first tackle the problem of learning, by implementing a more complex entangling structure where the ansatz varies layer by layer we can see whether the model "learns" more effecitvely despite a lack of access to parameters:

```
...as before

# train the model
classifier = ClassifierQNN(models.strongly_entangled_ansatz, enhanced_data, encoded_target, N_LAYERS)
classifier.train_test_split()

# score the model
classifier.learn_model(epochs=100)
classifier.score_model()
fig = classifier.plot_fit(show=True, decoded_data=encoded_data)

```

The resulting accuracy and cross entropy loss and accuracy is...

```
Cross entropy loss on training set: 0.32373467087745667
Accuracy of fullmodel on training set: 0.8975

Cross entropy loss on test set: 0.31316033005714417
Accuracy of fullmodel on test set: 0.91
```

![classification using complex ansatz](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classifier/strongly_entangling_ansatz/features=4&layers=5&epochs=100&noise=0.2.svg?raw=true)

This appears to be better, as the model is actually trying to minimise the number of erros and has some perspective of the classification problem, but clearly the model is limited somehow. This is likely due to a lack of available fourier frequencies, with which to assign fourier coefficents. Naturally a function which can only be decomposed into one fourier frequency lacks the complexity necessary to approximate an arbitrary multivariate function. [Schuld, M. 2020](https://doi.org/10.48550/arXiv.2008.08605) suggests that by reuploading the data via our encoding method we also increase fourier frequencies with which our model has access too. As follows:

```
...as before

# train the model
classifier = ClassifierQNN(models.data_reupload, enhanced_data, encoded_target, N_LAYERS)
classifier.train_test_split()

# score the model
classifier.learn_model(epochs=100)
classifier.score_model()
fig = classifier.plot_fit(show=True, decoded_data=encoded_data)

```

The resulting accuracy and cross entropy loss and accuracy is...

```
Cross entropy loss on training set: 0.32373467087745667
Accuracy of fullmodel on training set: 0.8975

Cross entropy loss on test set: 0.31316033005714417
Accuracy of fullmodel on test set: 0.91
```

![classification with data reupload](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classifier/data_reupload/features=4&layers=5&epochs=100&noise=0.2.svg?raw=true)

![classification with mid measure ](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classifier/mid_circuit/features=4&layers=5&epochs=100&noise=0.2.svg?raw=true)

Prior to running the model on a standardised classification library, metrics were utilised to predict the power of each model. Three metrics were utilised. 

The first follows [Schuld, M. 2020](https://doi.org/10.48550/arXiv.2008.08605), by representing the model a as a fourier series we uncover that the power of the model depends on the coefficents the model has access to control. A model with only a few adjustable fourier coefficents is naturally going to act as a weaker arbitrary function approximator.

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/compare_VQC_models.svg?raw=true)

With Convolutional network simple ansatz no data reupload

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/convolution_simple_ansatz.svg?raw=true)

With Convolutional network simple ansatz and data reupload

![Fourier Coefficents sampling on elementary VQC models](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/fourier/convolution_data_reupload.svg?raw=true)


Sample Fisher Information eigenvalue distributions

![Eigenvalue distribution of classical fisher information matrix](https://github.com/maxmarvell/qnn_non_linearity/blob/main/graphs/classical_fisher/compare_VQC_models.svg?raw=true)


## Outlooks

Beyond the work outlined here, it would be good to apply some noise simulations as we would likely find that larger models such as the mid-circuit measurements would drop off in utility. Moreover we could attempt to implement a quantum model where non-linearites are induced by the open quantum dynamics of a noisy system.

It would also be useful to impelement these models on a more complex learning task to really evaluate effectiveness.
