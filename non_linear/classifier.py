from functools import partial
import pennylane.numpy as np
import pennylane as qml
import jax.numpy as jnp
import jax
import optax
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from non_linear.utils import graph_utils

from non_linear.models import qnn_compiler
from non_linear.fourier import SampleFourierCoefficients
from non_linear.fisher import FisherInformation
from non_linear.qfisher import QuantumFisherInformation

from non_linear.utils.linked_list import NNLinkedList, LearnModelData
from non_linear.autoencoder import Autoencoder

from numpy import ndarray
import numpy as onp

# Class for classification tasks
class ClassifierQNN():

    def __init__(self, model: any, data: np.ndarray, target: np.ndarray, n_layers: int) -> None:

        # extract target length and number of features
        self.target_length = target.shape[1]
        self.n_features = data.shape[1]

        # store number of layers
        self.n_layers = n_layers
        
        # store other
        self.model = model
        self.data = data
        self.target = target

        # configure the compiler for classification
        self.compiler = qnn_compiler(model, self.n_features, n_layers, self.target_length)

        # extract the correct parameter shape
        self.parameter_shape = self.compiler.parameter_shape

        # batch the input to the classifier
        qnn = self.compiler.classification()
        batched = jax.vmap(qnn, (0, None))
        self.qnn = jax.jit(batched)

    def train_test_split(self, test_size:float=0.20):

        '''
            splits input into test and train set, interface to reduce complexity in number of params for __init__

            **kwargs:
                test_size: the proporion of samples that should be assigned to the test
        '''

        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test


    @partial(jax.jit, static_argnums=(0,))
    def cross_entropy_loss(self, y_true, y_pred):

        '''
            calculate cross entropy loss between prediction and true target
        '''

        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))

    @partial(jax.jit, static_argnums=(0,))
    def calculate_ce_cost(self, X, y, theta):

        # get prediction
        yp = jnp.array(self.qnn(X, theta)).T

        # softmax the output
        yp = jax.nn.softmax(yp)

        # compute loss metric between prediction and real
        cost = self.cross_entropy_loss(y, yp)

        return cost

    @partial(jax.jit, static_argnums=(0,))
    def optimizer_update(self, opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: self.calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    def learn_model(self, 
                    epochs: int, 
                    seed: int = random.randrange(1000), 
                    lr:float = 0.04,
                    save_data:bool = False):

        self.optimizer = optax.adam(learning_rate=lr)

        # set seed
        key = jax.random.PRNGKey(seed)

        # initialize circuit params
        initial_params = jax.random.normal(key, shape=self.parameter_shape)
        params = jnp.copy(initial_params)

        # initialize optimizer
        opt_state = self.optimizer.init(initial_params)

        # init empty list to store fishers, qfishers, etc
        self.training_data = []

        ##### FIT #####
        for epoch in range(epochs):
            params, opt_state, cost = self.optimizer_update(opt_state, params, self.X_train, self.y_train)
            if epoch % 5 == 0:
                print(f'epoch: {epoch}\t cost: {cost}')

            if (epoch % 25 == 0 and save_data):

                i = epoch // 25

                # fisher = FisherInformation(self.compiler)
                # fisher.fisher_information_matrix, fisher.fisher_eigenvalues = fisher.batched_fisher_information(self.X_train, params)

                # qfisher = QuantumFisherInformation(self.compiler)
                # qfisher.quantum_fisher_information_matrix, qfisher.quantum_fisher_eigenvalues = qfisher.batched_quantum_fisher_information(self.X_train, params)


                # y_pred = self.qnn(self.data, params)

                data = LearnModelData(
                    # quantum_fisher_information=qfisher,
                    # fisher_information=fisher,
                    params=params,
                    # y_pred=y_pred
                )

                data.epoch = epoch
                
                self.training_data.append(data)

        # store trained parameters
        self.fit_params = params


    def score_model(self):

        # Evaluate the cross entropy loss on the training set
        yp = jnp.array(self.qnn(self.X_train, self.fit_params)).T
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on training set: {self.cross_entropy_loss(self.y_train,yp)}')

        # Evaluate the accuracy on the training set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on training set: {accuracy_score(jnp.argmax(self.y_train,axis=1),yp)}\n')

        # Evaluate the Cross Entropy Loss on the testing set
        yp = jnp.array(self.qnn(self.X_test, self.fit_params)).T
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on test set: {self.cross_entropy_loss(self.y_test,yp)}')

        # Evaluate the accuracy on the testing set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on test set: {accuracy_score(jnp.argmax(self.y_test,axis=1),yp)}\n')


    def plot_fit(self, decoded_data:ndarray = None, show:bool = False):

        figure = plt.figure()
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(111)
        
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which='minor', linewidth=0.5, alpha=0.5)
        ax.set(xlabel=R'$x$', ylabel=R"$y$")
        plt.xlabel("x", size=14, fontname="Times New Roman", labelpad=10)
        plt.ylabel("y", size=14, fontname="Times New Roman", labelpad=10)

        # decode where necessary
        if type(decoded_data) == ndarray:
            X_train, X_test, _, _ = train_test_split(decoded_data, self.target, test_size=self.test_size, random_state=42)
        else:
            X_test = self.X_test
            X_train = self.X_train

        # Apply final params to test set
        yp_test = jnp.array(self.qnn(self.X_test, self.fit_params)).T
        yp_test = jax.nn.softmax(yp_test)
        yp_test = jnp.argmax(yp_test, axis=1)

        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=yp_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # Apply final params to test set
        yp_train = jnp.array(self.qnn(self.X_train, self.fit_params)).T
        yp_train = jax.nn.softmax(yp_train)
        yp_train = jnp.argmax(yp_train, axis=1)

        # Plot the testing points
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=yp_train,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )
        
        # Identify errors 
        errors = yp_test - jnp.argmax(self.y_test, axis=1)
        args = jnp.argwhere(errors != 0)
        
        ax.scatter(
            X_test[args, 0],
            X_test[args, 1],
            edgecolors="c",
            linewidths=1.8,
            s=120, 
            facecolors='none',
        )
        
        errors = yp_train - jnp.argmax(self.y_train, axis=1)
        args = jnp.argwhere(errors != 0)
        
        ax.scatter(
            X_train[args, 0],
            X_train[args, 1],
            edgecolors="c",
            linewidths=1.8,
            s=120, 
            facecolors='none',
        )
        
        plt.grid()

        if show: plt.show()

        return figure
    

    def fourier_coefficents(self, n_samples:int = 100, n_coeffs:int = 5, show:bool = False, ax = None):
        self.fourier.random_sample(n_coeffs, n_samples)
        return self.fourier.plot_coeffs(show, ax)

    

    def plot_fisher_histogram(self, show:bool = False, ax = None):
        
        assert self.training_data != None, "need to train the model first"

        # if no array of axes provided set a default
        if type(ax) != ndarray: 
            fig, ax = plt.subplots(1, len(self.training_data), figsize=(15,4), sharey=True)

        for state in self.training_data:

            fisher_matrices = jnp.empty(shape=(self.X_train.shape[0], np.product(self.parameter_shape), np.product(self.parameter_shape)))
            eigenvalues = jnp.empty(shape=(self.X_train.shape[0], np.product(self.parameter_shape,)))

            fisher = FisherInformation(self.compiler)
            values, grads = fisher.batched_fisher_information(self.X_train, state.params)

            values = onp.asarray(values)
            grads = onp.asarray(grads)

            for i in range(self.X_train.shape[0]):
                fisher_matrices = fisher_matrices.at[i].set(onp.sum(onp.outer(grads[i,j],grads[i,j])/values[i,j] for j in fisher.batch_index))
                eigenvalues = eigenvalues.at[i].set(onp.linalg.eigvals(fisher_matrices[i]))


            fisher.matrices = fisher_matrices
            fisher.eigenvalues = eigenvalues

            state.fisher_information = fisher

        for i, state in enumerate(self.training_data):
            ax[i].set_title(state.epoch)
            ax[i].boxplot(state.fisher_information.eigenvalues.reshape(-1))

        plt.show()


    def plot_quantum_fisher(self, show:bool = False, ax = None):

        assert self.training_data != None, "need to train the model first"

        # if no array of axes provided set a default
        if type(ax) != ndarray: 
            fig, ax = plt.subplots(1, len(self.training_data), figsize=(10,4))
        
        for state in self.training_data:
            qfisher = QuantumFisherInformation(self.compiler)
            qfisher.matrix, qfisher.eigenvalues = qfisher.batched_quantum_fisher_information(self.X_train, state.params)

            state.fisher_information = qfisher

        for i, state in enumerate(self.training_data):
            ax[i].set_title(state.epoch)
            ax[i].boxplot(state.fisher_information.eigenvalues.reshape(-1))
        
        plt.show()


