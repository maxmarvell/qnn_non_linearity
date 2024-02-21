from functools import partial
import matplotlib.pyplot as plt
from pennylane import numpy as np
import jax

# Defining a class that computes fisher information given a quantum model
class FisherInformation():

  def __init__(self, pqc_compiler, n_features, n_layers, n_samples):

    '''
      n_features: Number of features of the input data
      n_layers: Number of parametrised layers of the pqc
      n_samples: Number of samples of the input data
    '''

    # Initailise the quantum state and return the parameter shape
    self.qnn, self.parameter_shape = pqc_compiler(n_features, n_layers)
    self.n_samples = n_samples

    # Random Inputs
    self.inputs = np.random.normal(0, 1, (n_samples, n_features,))

    self.probability_index = np.array([i for i in range(n_features**2)])

    # Random initial parameters - but same for each sample
    self.params = np.tile(jax.random.uniform(jax.random.PRNGKey(10), shape=self.parameter_shape), (n_samples, 1, 1, 1))

  # This needs to batched here as value_and_grad cannot handle batched output
  @partial(jax.jit, static_argnums=(0,))
  def get_grads(self, inputs: np.ndarray, params: np.ndarray, i):
    return jax.value_and_grad(lambda theta: self.qnn(inputs, theta.reshape(self.parameter_shape))[i])(params.reshape(-1))

  # Batching the gradient computation of each probability output
  def get_fisher(self, inputs:np.ndarray, params:np.ndarray):
    grads_batched = jax.vmap(self.get_grads, (None, None, 0))
    get_grads = jax.jit(grads_batched)
    values, grads = get_grads(inputs, params, self.probability_index)
    return values, grads

  # Computing the FIM for each sample input
  def get_fishers(self):
    fishers_batched = jax.vmap(self.get_fisher)
    fisher = jax.jit(fishers_batched)
    value, grad = fisher(self.inputs, self.params)

    fishers = np.zeros(shape=(self.n_samples, np.product(self.parameter_shape), np.product(self.parameter_shape)))
    e = np.array([])
    for i in range(self.n_samples):
      fishers[i] = np.sum(np.outer(grad[i,j],grad[i,j])/value[i,j] for j in self.probability_index)
      e = np.append(e, np.linalg.eigvals(fishers[i]))

    self.fishers = fishers
    self.e = e

    return fishers, e

  def plot_eigenvalue_distribution(self):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Eigenvalue distribution of FIM')
    ax1.boxplot(self.e)
