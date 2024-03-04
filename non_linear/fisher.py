from functools import partial
import matplotlib.pyplot as plt
from pennylane import numpy as np
from numpy import ndarray
import jax

from non_linear.models import qnn_compiler

# Defining a class that computes fisher information given a quantum model
class FisherInformation():

  def __init__(self, model:any, n_features:int, n_layers:int):

    '''
      n_features: Number of features of the input data
      n_layers: Number of parametrised layers of the pqc
      n_samples: Number of samples of the input data
    '''

    # initailise the quantum state and return the parameter shape
    compiler = qnn_compiler(model, n_features, n_layers, 1)
    self.parameter_shape = compiler.parameter_shape
    self.qnn = compiler.probabilites()
    self.n_features = n_features

  # this needs to batched here as value_and_grad cannot handle batched output
  @partial(jax.jit, static_argnums=(0,))
  def get_grads(self, inputs: np.ndarray, params: np.ndarray, i):
    return jax.value_and_grad(lambda theta: self.qnn(inputs, theta.reshape(self.parameter_shape))[i])(params.reshape(-1))

  # batching the gradient computation of each probability output
  def get_fisher(self, inputs:np.ndarray, params:np.ndarray):
    grads_batched = jax.vmap(self.get_grads, (None, None, 0))
    get_grads = jax.jit(grads_batched)
    values, grads = get_grads(inputs, params, self.probability_index)
    return values, grads

  # computing the FIM for each sample input
  def sample_fishers(self, n_samples:int = 100):

    # random inputs
    self.inputs = np.random.normal(0, 1, (n_samples, self.n_features,))

    # indexes to batch with
    self.probability_index = np.array([i for i in range(self.n_features**2)])

    # random initial parameters - but same for each sample
    self.params = np.tile(jax.random.uniform(jax.random.PRNGKey(10), shape=self.parameter_shape), (n_samples, 1, 1, 1))

    # batch the fishers
    fishers_batched = jax.vmap(self.get_fisher)
    fisher = jax.jit(fishers_batched)
    value, grad = fisher(self.inputs, self.params)

    # init fisher information matrix and eigenvalues
    fishers = np.zeros(shape=(n_samples, np.product(self.parameter_shape), np.product(self.parameter_shape)))
    e = np.array([])

    # loop over all samples
    for i in range(n_samples):
      fishers[i] = np.sum(np.outer(grad[i,j],grad[i,j])/value[i,j] for j in self.probability_index)
      e = np.append(e, np.linalg.eigvals(fishers[i]))

    # assign fisher information matrices and eigenvalues
    self.fishers = fishers
    self.e = e

    return fishers, e

  def plot_eigenvalue_distribution(self, show:bool = None, ax = None):

    # if no array of axes provided set a default
    if ax == None: 
        fig, ax = plt.subplots(1, 1, figsize=(4,8))
        plt.tight_layout(pad=0.5)

    ax.boxplot(self.e)

    if show: plt.show()

    if ax == None: return fig, ax
