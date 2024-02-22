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
import utils.graph_utils

from models import qnn_compiler

from numpy import ndarray

class QCNN():

    def __init__(self, model:any, data:np.ndarray, target:np.ndarray, n_layers:int, nn_layers:tuple=(2,2,1), method:str="REG"):
        
        '''
            args:
                model:
                data:
                target:
                n_layers:
        
            **kwargs:
                nn_layers:  structure of the hybrid classical qunatum convolutional neural network
                method:     parametrisation method, "REG" only parametrise quantum circuit, "ACAF" parametrise classical non-linear activation function
        '''
        
        
        self.target_length = target.shape[1]
        self.n_features = data.shape[1]
        
        self.target = target
        self.data = data
    
        self.n_layers = n_layers
        self.nn_layers = nn_layers
        self.model = model
        self.method = method

        count = 0
        parameter_shapes = []

        for i, v in enumerate(self.nn_layers):
            if i == 0:
                param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_features)
            else:
                param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.nn_layers[i-1])
                
            if method == "ACAF":
                count += v
                
            count += v * np.product(param_shape)
            parameter_shapes.append((v, *param_shape))
            
        self.parameter_shapes = parameter_shapes
        self.parameter_count = count
    

    def train_test_split(self, test_size:float=0.20):
        res = train_test_split(self.data, self.target, test_size=0.20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = res
        return res


    @partial(jax.jit, static_argnums=(0))
    def forward_pass(self, inputs:ndarray, params:ndarray):
        
        depth = 0
        
        for i, layer in enumerate(self.nn_layers):
            
            if i == 0:
                # First layer
                compiler = qnn_compiler(self.model, self.n_features, self.n_layers, 1)
            elif i == len(self.nn_layers) - 1:
                # Output layer
                compiler = qnn_compiler(self.model, self.nn_layers[i-1], self.n_layers, self.target_length)
            else:
                # Hidden layers
                compiler = qnn_compiler(self.model, self.nn_layers[i-1], self.n_layers, 1)
                
            # Here we vmap params not input to process multiple nodes in the same layer
            qnn = compiler.classification()
            qnn_batched = jax.vmap(qnn, (None, 0))
            qnn = jax.jit(qnn_batched)
            
            parameter_shape = self.parameter_shapes[i]
            
            # Get the num parameters for the first layers
            num_params = np.product(parameter_shape)
            
            inputs = qnn(inputs, params[depth:depth+num_params].reshape(parameter_shape))
            
            if self.method == "ACAF":
                inputs = (1 / (1 + jnp.exp(-inputs))) * params[depth+num_params:depth+num_params+layer]
                depth += layer
            
            depth += num_params

        return inputs
    

    @partial(jax.jit, static_argnums=(0))
    def batched(self, inputs, params):
        forward_pass = jax.vmap(self.forward_pass, (0, None))
        forward_batched = jax.jit(forward_pass)
        return forward_batched(inputs, params)  
    

    @partial(jax.jit, static_argnums=(0,))
    def cross_entropy_loss(self, y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    

    @partial(jax.jit, static_argnums=(0,))
    def calculate_ce_cost(self, X, y, theta):

        # Get the target prediction
        yp = self.batched(X, theta).reshape(-1, self.target_length)

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
        
        self.optimizer = optax.adam(learning_rate=0.04)
        
        # Seed
        key = jax.random.PRNGKey(seed)

        # Initialize circuit params
        initial_params = jax.random.normal(key, shape=(self.parameter_count,))
        params = jnp.copy(initial_params)
        
        # Initialize optimizer
        opt_state = self.optimizer.init(initial_params)


        ##### FIT #####
        for epoch in range(epochs):
            params, opt_state, cost = self.optimizer_update(opt_state, params)
            if epoch % 5 == 0:
                print(f'epoch: {epoch}\t cost: {cost}')

            if epoch % 10 == 0:
                pass

        # Store trained parameters
        self.fit_params = params
    
    
    def batched(self, inputs, params):

        batched_passes = jax.vmap(self.forward_pass, (0, None))
        forward = jax.jit(batched_passes)

        return forward(inputs, params)
    
       
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