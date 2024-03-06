from functools import partial
import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax
import optax
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import non_linear.utils.graph_utils

from non_linear.models import qnn_compiler
from non_linear.fourier import SampleFourierCoefficients
from non_linear.fisher import FisherInformation
from non_linear.qfisher import QuantumFisherInformation
from non_linear.utils.linked_list import NNLinkedList, Node

from numpy import ndarray
from copy import deepcopy

class QCNN():

    def __init__(self, model:any, data:np.ndarray, target:np.ndarray, n_layers:int, nn_layers:tuple=(2,2,1), method:str="REG"):
        
        '''
            args:
                model: a model for a variational quantum circuit to implement
                data: the dataset
                target: the target variable
                n_layers: the number of layers of each qnode
        
            **kwargs:
                nn_layers:  structure of the hybrid classical qunatum convolutional neural network
                method:     parametrisation method, "REG" only parametrise quantum circuit, "ACAF" parametrise classical non-linear activation function
        '''
        
        # extract the target length and the number of features required
        self.target_length = target.shape[1]
        self.n_features = data.shape[1]
        
        # store classification data
        self.target = target
        self.data = data

        # store model parameters to class
        self.n_layers = n_layers
        self.nn_layers = nn_layers
        self.model = model
        self.method = method

        # initialise linked list to store fisher informations
        self.llist = NNLinkedList()

        # init parameter count
        count = 0
        parameter_shapes = []

        # build out parameter shape for the convolutional network
        for i, v in enumerate(self.nn_layers):
            if i == 0:
                param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_features)
            else:
                param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.nn_layers[i-1])
                
            if method == "ACAF":
                count += v
                
            count += v * np.product(param_shape)
            parameter_shapes.append((v, *param_shape))
        
        # store parameter quantities
        self.parameter_shapes = parameter_shapes
        self.parameter_count = count

    def train_test_split(self, test_size:float=0.20):

        '''
            splits input into test and train set, interface to reduce complexity in number of params for __init__

            **kwargs:
                test_size: the proporion of samples that should be assigned to the test
        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    @partial(jax.jit, static_argnums=(0,3))
    def forward_pass(self, 
                     inputs:ndarray, 
                     params:ndarray, 
                     compute_metrics:bool = False):
        
        '''
            computes a complete single pass of the entire convolutional neural network for a single sample
            optionally computes fisher information and/or quantum fisher information at each node.

            args:
                inputs: first layer of inputs
                params: full set of trainable parameters for entire network
            **kwargs:
                compute_metrics: if true computes the fisher information matrix and quantum fisher information of each node
        '''
        
        # first layer
        depth = 0
        pointer_node = None
        
        # loop over layers to build full model (this can be vmapped this way)
        for i, layer in enumerate(self.nn_layers):

            # init node for this layer
            new_node = Node(layer=i)

            if (pointer_node != None): pointer_node.next = new_node
            else: self.llist.head = new_node
            
            # logic to determine compiler target and wire length
            if i == 0:
                # first layer
                compiler = qnn_compiler(self.model, self.n_features, self.n_layers, 1)
            elif i == len(self.nn_layers) - 1:
                # output layer
                compiler = qnn_compiler(self.model, self.nn_layers[i-1], self.n_layers, self.target_length)
            else:
                # hidden layers
                compiler = qnn_compiler(self.model, self.nn_layers[i-1], self.n_layers, 1)
                
            # vmap params not input to process multiple nodes in the same layer
            qnn = compiler.classification()
            qnn_batched = jax.vmap(qnn, (None, 0))
            qnn = jax.jit(qnn_batched)
            
            # get num parameters
            parameter_shape = self.parameter_shapes[i]
            num_params = np.product(parameter_shape)

            # logic to compute fisher information for given layer where true
            if self.compute_metrics:
                fisher = FisherInformation(compiler)
                fisher.fisher_information_matrix = fisher.batched_fisher_information(inputs, params[depth:depth+num_params].reshape(parameter_shape))
                new_node.fisher_information = fisher

                qfisher = QuantumFisherInformation(compiler)
                qfisher.quantum_fisher_information_matrix = qfisher.batched_quantum_fisher_information(inputs, params[depth:depth+num_params].reshape(parameter_shape))
                new_node.quantum_fisher_information = qfisher


            # outpus of current layer -> inputs of next layer
            inputs = qnn(inputs, params[depth:depth+num_params].reshape(parameter_shape))
            new_node.outputs = inputs

            # move pointer
            pointer_node = new_node
            
            # increase depth
            depth += num_params

            # conditionally apply classical activation function
            if self.method == "ACAF":
                inputs = (1 / (1 + jnp.exp(-inputs))) * params[depth:depth+layer]
                depth += layer
            
        return inputs
    

    @partial(jax.jit, static_argnums=(0,3))
    def batched_forward_pass(self, inputs, params, compute_metrics:bool = False):

        '''
            computes the single forward pass for many inputs and a single parametrisation

            args:
                inputs: first layer of inputs
                params: full set of trainable parameters for entire network
            **kwargs:
                compute_fisher: if true computes the fisher information matrix of each node of each sample
                compute_qfisher: if true computes the quantum fisher information matrix at each node
        '''

        forward_pass = jax.vmap(self.forward_pass, (0, None, None))
        forward_batched = jax.jit(forward_pass)
        return forward_batched(inputs, params, compute_metrics)  
    

    @partial(jax.jit, static_argnums=(0,))
    def cross_entropy_loss(self, y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    

    @partial(jax.jit, static_argnums=(0,))
    def calculate_ce_cost(self, X, y, theta):

        batched = jnp.array(self.batched_forward_pass(X, theta))

        # Get the target prediction
        yp = batched.reshape(-1, self.target_length)

        # Softmax the output
        yp = jax.nn.softmax(yp)

        # Compute loss metric between prediction and real
        cost = self.cross_entropy_loss(y, yp)
        return cost
   
    
    @partial(jax.jit, static_argnums=(0,))
    def optimizer_update(self, opt_state, params):
        loss, grads = jax.value_and_grad(lambda theta: self.calculate_ce_cost(self.X_train, self.y_train, theta))(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    
    def learn_model(self, epochs: int, seed: int = random.randrange(1000)):

        '''
            train the model to obtain a final fit params
        '''
        
        self.optimizer = optax.adam(learning_rate=0.04)
        
        # seed
        key = jax.random.PRNGKey(seed)

        # initialize circuit params
        initial_params = jax.random.normal(key, shape=(self.parameter_count,))
        params = jnp.copy(initial_params)
        
        # initialize optimizer
        opt_state = self.optimizer.init(initial_params)

        # initialse a list of linked lists to store all the associated data (only sample every 25 epochs)
        self.training_data = [NNLinkedList() for _ in range(epochs // 25)]

        self.compute_metrics=False


        ##### FIT #####
        for epoch in range(epochs):
            params, opt_state, cost = self.optimizer_update(opt_state, params)

            if epoch % 5 == 0:
                print(f'epoch: {epoch}\t cost: {cost}')

            if epoch % 25 == 0:
                i = epoch // 25
                self.compute_metrics=True
                self.batched_forward_pass(self.X_train, params, compute_metrics=True)
                self.training_data[i] = self.llist
                self.compute_metrics=False

        # store trained parameters
        self.fit_params = params
    
       
    def score_model(self):

        # Evaluate the cross entropy loss on the training set
        yp = self.batched(self.X_train, self.fit_params).reshape(-1, self.target_length)
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on training set: {self.cross_entropy_loss(self.y_train,yp)}')

        # Evaluate the accuracy on the training set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on training set: {accuracy_score(jnp.argmax(self.y_train,axis=1),yp)}\n')

        # Evaluate the Cross Entropy Loss on the testing set
        yp = self.batched(self.X_test, self.fit_params).reshape(-1, self.target_length)
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on test set: {self.cross_entropy_loss(self.y_test,yp)}')

        # Evaluate the accuracy on the testing set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on test set: {accuracy_score(jnp.argmax(self.y_test,axis=1),yp)}\n')


    def plot_fit(self):
        
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

        # Apply final params to test set
        yp_test = self.batched(self.X_test, self.fit_params).reshape(-1, self.target_length)
        yp_test = jax.nn.softmax(yp_test)
        yp_test = jnp.argmax(yp_test, axis=1)

        # Plot the testing points
        ax.scatter(
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=yp_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # Apply final params to test set
        yp_train = self.batched(self.X_train, self.fit_params).reshape(-1, self.target_length)
        yp_train = jax.nn.softmax(yp_train)
        yp_train = jnp.argmax(yp_train, axis=1)

        # Plot the testing points
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=yp_train,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )
        
        # Identify errors 
        errors = yp_test - jnp.argmax(self.y_test, axis=1)
        args = jnp.argwhere(errors != 0)
        
        ax.scatter(
            self.X_test[args, 0],
            self.X_test[args, 1],
            edgecolors="c",
            linewidths=1.8,
            s=120, 
            facecolors='none',
        )
        
        errors = yp_train - jnp.argmax(self.y_train, axis=1)
        args = jnp.argwhere(errors != 0)
        
        ax.scatter(
            self.X_train[args, 0],
            self.X_train[args, 1],
            edgecolors="c",
            linewidths=1.8,
            s=120, 
            facecolors='none',
        )
        
        plt.grid()

        return figure
    
    def plot_fishers(self):
        pass

    def plot_qfishers(self):
        pass
    
    def fourier_coefficents(self, n_coeffs:int = 5, n_samples:int = 100, show:bool = False, ax = None):
        self.target_length = 1
        fourier = SampleFourierCoefficients(self.batched, parameter_shape=(self.parameter_count, ),  n_features=self.n_features)
        fourier.random_sample(n_coeffs, n_samples)
        self.target_length = self.target.shape[1]
        return fourier.plot_coeffs(show, ax)