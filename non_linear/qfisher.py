from itertools import product
from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from non_linear.models import qnn_compiler

class QuantumFisherInformation():

    def __init__(self, model:qnn_compiler) -> None:

        '''
            initializes a class that can compute quantum fisher information in a batched or
            non-batched fashion

            qnn_compiler: comiler to create the variational quantum circuit model
        '''

        # extract parameter shape
        self.parameter_shape = model.parameter_shape

        # initailise the quantum circuit for state measurement
        self.qnn = model.state()

        # extract useful params for fidelity calcs
        self.n_features = self.parameter_shape[-2]
        self.wires = [i for i in range(self.n_features)]


    @partial(jax.jit, static_argnums=(0,))
    def compute_fidelity(self, inputs, params, index):

        # unpack the index
        i, j = index

        # (e_i + e_j) term
        shifted = params.copy()
        shifted = shifted.at[i].add(np.pi/2)
        shifted = shifted.at[j].add(np.pi/2)
        term1 = qml.qinfo.fidelity(self.qnn, self.qnn, wires0=self.wires, wires1=self.wires)(
            (inputs, params.reshape(self.parameter_shape)), (inputs, shifted.reshape(self.parameter_shape)))

        # (e_i - e_j) term
        shifted = params.copy()
        shifted = shifted.at[i].add(-np.pi/2)
        shifted = shifted.at[j].add(np.pi/2)
        term2 = qml.qinfo.fidelity(self.qnn, self.qnn, wires0=self.wires, wires1=self.wires)(
        (inputs, params.reshape(self.parameter_shape)), (inputs, shifted.reshape(self.parameter_shape)))

        # (-e_i + e_j) term
        shifted = params.copy()
        shifted = shifted.at[i].add(np.pi/2)
        shifted = shifted.at[j].add(-np.pi/2)
        term3 = qml.qinfo.fidelity(self.qnn, self.qnn, wires0=self.wires, wires1=self.wires)(
        (inputs, params.reshape(self.parameter_shape)), (inputs, shifted.reshape(self.parameter_shape)))

        # (-e_i - e_j) term
        shifted = params.copy()
        shifted = shifted.at[i].add(-np.pi/2)
        shifted = shifted.at[j].add(-np.pi/2)
        term4 = qml.qinfo.fidelity(self.qnn, self.qnn, wires0=self.wires, wires1=self.wires)(
        (inputs, params.reshape(self.parameter_shape)), (inputs, shifted.reshape(self.parameter_shape)))

        return -1/2 * (term1 - term2 - term3 + term4)

    @partial(jax.jit, static_argnums=(0,))
    def quantum_fisher_information(self, inputs, params):

        # using itertools to get all permutations and vmap the calculation of QFI
        indexes = jnp.array(list(product(range(np.prod(self.parameter_shape)), repeat=2)))

        # batch the fidelity count
        batched = jax.vmap(self.compute_fidelity, (None, None, 0))
        fisher_matrix = jax.jit(batched)(inputs.reshape(-1), params.reshape(-1), indexes).reshape(np.prod(self.parameter_shape),np.prod(self.parameter_shape))

        # expected shape of eigenvalues
        result_shape = jax.core.ShapedArray((params.reshape(-1).shape[0],), 'float32')

        # function for pure callback
        @jax.jit
        def f(x):
            jnp.linalg.eigvals(x)

        # yield eigenvalues
        e = jax.pure_callback(f, result_shape, fisher_matrix)

        return fisher_matrix, e
  

    @partial(jax.jit, static_argnums=(0,))
    def batched_quantum_fisher_information(self, inputs, params):

        # using itertools to get all permutations and vmap the calculation of QFI
        indexes = jnp.array(list(product(range(np.prod(self.parameter_shape)), repeat=2)))

        batched_gradients = jax.vmap(self.compute_fidelity, (None, None, 0))
        f = lambda inputs, params: batched_gradients(inputs, params, indexes).reshape(np.prod(self.parameter_shape),np.prod(self.parameter_shape))

        batched = jax.vmap(f, (0, None))
        quantum_fisher_matrices = jax.jit(batched)(inputs, params)

        # expected shape of eigenvalues
        result_shape = jax.core.ShapedArray((params.reshape(-1).shape[0],), 'float32')

        # function for pure callback
        @jax.jit
        def f(x):
            jnp.linalg.eigvals(x)

        eigenvalues = jnp.empty(shape=(inputs.shape[0],np.prod(self.parameter_shape),))
        for i in range(inputs.shape[0]):
            eigenvalues = eigenvalues.at[i].set(jnp.linalg.eigvals(quantum_fisher_matrices[i]))

        return quantum_fisher_matrices, eigenvalues


    @partial(jax.jit, static_argnums=(0,))
    def sample_quantum_fisher_information(self, n_samples):

        self.n_samples = n_samples

        # random inputs
        self.inputs = np.random.normal(0, 1, (n_samples, self.n_features,))

        # random initial parameters - but same for each sample
        self.params = jax.random.uniform(jax.random.PRNGKey(0), shape=self.parameter_shape)

        # batch the function only over the random inputs
        fishers_batched = jax.vmap(self.quantum_fisher_information, (0, None))
        fishers = jax.jit(fishers_batched)

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
