from non_linear.models import qnn_compiler, simple_ansatz
import pennylane.numpy as np
import jax

N_FEATURES = 4
N_TARGETS = 2
N_LAYERS = 5

# init the compiler
compiler = qnn_compiler(simple_ansatz, N_FEATURES, N_LAYERS, N_TARGETS)

# this model is for classification
qnn = compiler.classification()

inputs = np.array([.1,.2,.3,.4])
params = jax.random.uniform(jax.random.PRNGKey(0), shape=compiler.parameter_shape)

# this will yield an array of shape (2,) which can be used to classify the input
output = qnn(inputs, params)

print(output)