import jax
import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
from functools import partial


class QCNN():

  def __init__(self, qnn_compiler, num_features, num_layers, nn_layers=(2,2,1)):
    self.qnn_compiler = qnn_compiler
    self.num_features = num_features
    self.num_layers = num_layers
    self.nn_layers = nn_layers

  def estimate_required_parameters(self):
    count = 0
    parameter_shapes = []

    for i, v in enumerate(self.nn_layers):
      if i == 0:
        param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_features)
        count += v * np.product(param_shape)
        parameter_shapes.append((v, *param_shape))
      else:
        param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.nn_layers[i-1]*2)
        count += v * np.product(param_shape)
        parameter_shapes.append((v, *param_shape))

    self.parameter_shapes = parameter_shapes

    return count


  @partial(jax.jit, static_argnums=(0))
  def forward_pass(self, inputs, params):

    ### FIRST LAYER ###

    depth = 0

    # Extract details about the parameter shape of the layer i
    layer_parameters = self.parameter_shapes[0]
    n_features, n_layers = layer_parameters[2], layer_parameters[1]

    # Return the compiled qnn given the parameter shape of this layer
    qnn_temp, _ = self.qnn_compiler(n_features, n_layers)

    # Apply vmap on x (first circuit param)
    qnn_batched = jax.vmap(qnn_temp, (0, None))

    # Jit for faster execution
    qnn = jax.jit(qnn_batched)

    # Get the num parameters for the first layers
    num_params = np.product(layer_parameters)

    res = qnn(inputs, params[depth:depth+num_params].reshape(layer_parameters))

    ### SECOND LAYER ###

    depth += num_params

    # Extract details about the parameter shape of the layer i
    layer_parameters = self.parameter_shapes[1]
    n_features, n_layers = layer_parameters[2], layer_parameters[1]

    # Return the compiled qnn given the parameter shape of this layer
    qnn_temp, _ = self.qnn_compiler(n_features, n_layers)

    # Apply vmap on x (first circuit param)
    qnn_batched = jax.vmap(qnn_temp, (0, None))

    # Jit for faster execution
    qnn = jax.jit(qnn_batched)

    # Get the num parameters for the first layers
    num_params = np.product(layer_parameters)

    res = qnn(inputs, params[depth:depth+num_params].reshape(layer_parameters))


    ### THIRD LAYER ###

    depth += num_params

    # Extract details about the parameter shape of the layer i
    layer_parameters = self.parameter_shapes[2]
    n_features, n_layers = layer_parameters[2], layer_parameters[1]

    # Return the compiled qnn given the parameter shape of this layer
    qnn_temp, _ = self.qnn_compiler(n_features, n_layers)

    # Apply vmap on x (first circuit param)
    qnn_batched = jax.vmap(qnn_temp, (0, None))

    # Jit for faster execution
    qnn = jax.jit(qnn_batched)

    # Get the num parameters for the first layers
    num_params = np.product(layer_parameters)

    res = qnn(inputs, params[depth:depth+num_params].reshape(layer_parameters))

    print(res)

    return jnp.array(res)

  def batched(self, inputs, params):

    print(inputs.shape, params.shape)

    batched_passes = jax.vmap(self.forward_pass, (0, None))
    forward = jax.jit(batched_passes)

    return forward(inputs, params, i=0, depth=0)