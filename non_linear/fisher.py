from functools import partial
import matplotlib.pyplot as plt
from pennylane import numpy as np
import jax
import jax.numpy as jnp

from numpy import ndarray
import numpy as onp

from non_linear.models import qnn_compiler

# Defining a class that computes fisher information given a quantum model
class FisherInformation():

    def __init__(self, model:qnn_compiler):

        '''
            initializes a class that can compute classical fisher information in a batched or
            non-batched fashion

            args:
                qnn_compiler: comiler to create the variational quantum circuit model
        '''

        # extract parameter shape
        self.parameter_shape = model.parameter_shape

        # initailise the quantum circuit for state measurement
        self.qnn = model.probabilites()
        
        # extract useful params for fidelity calcs
        self.n_features = self.parameter_shape[-2]

        # utility index for batching gradient calcs
        self.batch_index = jnp.array([i for i in range(self.n_features**2)])

    @partial(jax.jit, static_argnums=(0,))
    def get_gradient(self, inputs:ndarray, params:ndarray, i:int):
        '''
            function to compute a gradient of a single bitstring outcome probability

            args:
                inputs: a numpy array containing one sample
                params: a numpy array containing a single parametrisation of the qnn
                i: the index of the bitstring outcompe we want to compute the probability gradient
        '''
        return jax.value_and_grad(lambda theta: self.qnn(inputs, theta.reshape(self.parameter_shape))[i])(params.reshape(-1))

    @partial(jax.jit, static_argnums=(0,))
    def batched_gradient(self, inputs:ndarray, params:ndarray):

        '''
            function to batch the gradient compute over all the possible bitstring outcomes, there are 2^(n_features) possible states

            args:
                inputs: a numpy array containing one sample
                params: a numpy array containing a single parametrisation of the qnn
        '''

        batched = jax.vmap(self.get_gradient, (None, None, 0))
        get_gradient = jax.jit(batched)
        values, grads = get_gradient(inputs, params, self.batch_index)
        return values, grads
  
    @partial(jax.jit, static_argnums=(0,))
    def fisher_information(self, input:ndarray, params:ndarray):

        '''
            function that interfaces the batched gradient to compute a single fisher information instance
            WARNING this func cannot be batched over inputs

            args:
                input: a numpy array containing one sample
                params: a numpy array containing a single parametrisation of the qnn
        '''

        value, grad = self.batched_gradient(input, params)

        result_shape_dtype = jax.ShapeDtypeStruct(
            shape=(params.reshape(-1).shape[0], params.reshape(-1).shape[0], ),
            dtype=value.dtype
        )

        f = lambda value, grad: jnp.sum(jnp.outer(grad[j],grad[j])/value[j] for j in self.batch_index)

        fisher_matrix = jax.pure_callback(f, result_shape_dtype, value, grad)

        eignevalue_shape_dtype = jax.ShapeDtypeStruct(
            shape=(params.reshape(-1).shape[0],),
            dtype=value.dtype
        )

        g = lambda x: jnp.linalg.eigvals(x)

        eigenvalues = jax.pure_callback(g, eignevalue_shape_dtype, fisher_matrix)
        
        return fisher_matrix, eigenvalues
    
    @partial(jax.jit, static_argnums=(0,))
    def batched_fisher_information(self, inputs:ndarray, params:ndarray):

        '''
            function that batches over self.fisher_information to compute the matrix for many samples
            implemented using pure callbacks to compute fisher matrices and eigenvalues

            DISCLAIMER pure callbacks increase runtime however only utilised for post-processing here

            args:
                inputs: a numpy array containing many samples
                params: a numpy array containing a single parametrisation of the qnn

            returns:
                matrices: a (n, m, m) array with the fisher information matrix for each sample
                eignevalues: a (n, m) array with the eigenvalues for each sample
        '''

        batched = jax.vmap(self.batched_gradient, (0, None))
        values, grads = jax.jit(batched)(inputs, params)

        # specify the fisher shape and dtype of the eigenvalues
        fisher_shape_dtype = jax.ShapeDtypeStruct(
            shape=(inputs.shape[0], params.reshape(-1).shape[0], params.reshape(-1).shape[0], ),
            dtype=values.dtype
        )

        # numpy function for pure callback
        def compute_fisher(values, grads):
            matrices = jnp.empty(shape=(inputs.shape[0], np.product(self.parameter_shape), np.product(self.parameter_shape)))
            for i in range(inputs.shape[0]):
                matrices = matrices.at[i].set(onp.sum(onp.outer(grads[i,j],grads[i,j])/values[i,j] for j in self.batch_index))
            return matrices
        
        # compute the fisher matrices using pure callback method
        matrices = jax.pure_callback(compute_fisher, fisher_shape_dtype, values, grads)

        # specify the ouput shape and dtype of the eigenvalues
        eignevalue_shape_dtype = jax.ShapeDtypeStruct(
            shape=(inputs.shape[0], params.reshape(-1).shape[0],),
            dtype=values.dtype
        )


        def f(matrices):
            eigenvalues = jnp.empty(shape=(inputs.shape[0], np.product(self.parameter_shape,)))
            for i in range(inputs.shape[0]):
                eigenvalues = eigenvalues.at[i].set(onp.linalg.eigvals(matrices[i]))
            return eigenvalues
        
        eigenvalues = jax.pure_callback(f, eignevalue_shape_dtype, matrices)

        return matrices, eigenvalues


    def sample_fisher_information_matrix(self, n_samples:int = 100):

        '''
            function that takes n samples of the fisher information of the model

            **kwargs:
                n_samples: an integer representing the number of samples to take
        '''

        # random inputs
        self.inputs = np.random.normal(0, 1, (n_samples, self.n_features,))

        # random initial parameters - but same for each sample
        self.params = jax.random.uniform(jax.random.PRNGKey(10), shape=self.parameter_shape)

        # batch the fishers
        batched = jax.vmap(self.fisher_information, (0, None,))
        fishers = jax.jit(batched)

        # process all the samples
        res = fishers(self.inputs, self.params)
        self.fisher_matrices, self.eigenvalues = res

        return res

    def plot_eigenvalue_distribution(self, show:bool = None, ax = None):

        # if no array of axes provided set a default
        if ax == None: 
            fig, ax = plt.subplots(1, 1, figsize=(4,8))
            plt.tight_layout(pad=0.5)

        ax.boxplot(self.eigenvalues)

        if show: plt.show()

        if ax == None: return fig, ax
