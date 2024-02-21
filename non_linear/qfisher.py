from itertools import product
from functools import partial
import jax
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

class QuantumFisherInformation():

  def __init__(self, pqc_compiler, n_features, n_layers, n_samples) -> None:

    '''
      n_features: Number of features of the input data
      n_layers: Number of parametrised layers of the pqc
      n_samples: Number of samples of the input data
    '''

    # Initailise the quantum state and return the parameter shape
    self.qnn, self.parameter_shape = pqc_compiler(n_features, n_layers)
    self.n_samples = n_samples
    self.wires = [i for i in range(n_features)]

    # Random Inputs
    self.inputs = np.random.normal(0, 1, (n_samples, n_features,))

    # Random initial parameters - but same for each sample
    self.params = np.tile(jax.random.uniform(jax.random.PRNGKey(0), shape=self.parameter_shape), (n_samples, 1, 1, 1))


  @partial(jax.jit, static_argnums=(0,))
  def compute_fidelity(self, inputs, params, index):

    # Unpack the index
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
  def compute_QFI(self, inputs, params):
    # Using itertools to get all permutations and vmap the calculation of QFI
    indexes = np.array(list(product(range(np.prod(self.parameter_shape)), repeat=2)))

    # Batch the function
    fidelity_batched = jax.vmap(self.compute_fidelity, (None, None, 0))
    fidelities = jax.jit(fidelity_batched)

    # Evaluate the QFI
    fisher = fidelities(inputs.reshape(-1), params.reshape(-1), indexes).reshape(np.prod(self.parameter_shape),np.prod(self.parameter_shape))

    return fisher

  def sample_QFI(self):
    # Batch the function
    fishers_batched = jax.vmap(self.compute_QFI)
    fishers = jax.jit(fishers_batched)

    # Get all FIM samples
    FIM = fishers(self.inputs, self.params)
    self.FIM = FIM
    return FIM

  def sample_eigenvalues(self):
    e = np.array([])
    for i in range(10):
      e = np.append(e, np.linalg.eigvals(self.FIM[i]))

    self.e = e
    return e

  def plot_eigenvalue_distribution(self):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Eigenvalue distribution of QFIM')
    ax1.boxplot(self.e)
