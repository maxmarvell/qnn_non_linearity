import models
from autoencoder import Autoencoder
from classifier import ClassifierQNN

from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pennylane as qml
import jax
import torch

# Load the make moons data set
data, target = make_moons(noise=.2)

# Encode the target using One Hot Encoder module
scalerOHE = OneHotEncoder(sparse_output=False)
encoded_target = scalerOHE.fit_transform(target.reshape(-1,1))
  
# Encode the input using a Scaler
sScaler = StandardScaler()
encoded_data = sScaler.fit_transform(data)

# Encode the data
autoencoder = Autoencoder(encoded_data, 10)
encoded_data = autoencoder.encoder(torch.from_numpy(encoded_data).float()).detach().numpy()

# Choose model
model, parameter_shape = models.data_reupload(10, 8)

dev = qml.device("default.qubit", wires=10)

@qml.qnode(dev)
def circuit(x, w):
    model(x, w)
    return qml.expval(qml.PauliZ(wires=0))

# Seed
key = jax.random.PRNGKey(42)

# Initialize circuit params
params = jax.random.normal(key, shape=parameter_shape)

res = qml.fourier.circuit_spectrum(circuit)(encoded_data, params)

print(res)

for inp, freqs in res.items():
    print(f"{inp}: {freqs}")

# classifier = ClassifierQNN(model, encoded_data, encoded_target, 4)
# classifier.train_test_split()
# classifier.learn_model(epochs=100)
# classifier.score_model()
# classifier.plot_fit()