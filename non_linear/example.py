NOISE = .2
EPOCHS = 100
N_SAMPLES = 500
N_LAYERS = 3
N_FEATURES = 5

### CLASSIFICATION HYBRID CALSSICAL QUANTUM ###

# import models
# from autoencoder import Autoencoder
# from classifier import ClassifierQNN
# from sklearn.datasets import make_moons
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# import torch


# # Load the make moons data set
# data, target = make_moons(n_samples=N_SAMPLES,noise=NOISE)

# # Encode the target using One Hot Encoder module
# scalerOHE = OneHotEncoder(sparse_output=False)
# encoded_target = scalerOHE.fit_transform(target.reshape(-1,1))
  
# # Encode the input using a Scaler
# sScaler = StandardScaler()
# encoded_data = sScaler.fit_transform(data)

# # # Encode the data
# # autoencoder = Autoencoder(encoded_data, N_FEATURES)
# # encoded_data = autoencoder.encoder(torch.from_numpy(encoded_data).float()).detach().numpy()

# # Choose model
# model = models.data_reupload

# # Train the model
# classifier = ClassifierQNN(model, encoded_data, encoded_target, N_LAYERS)

# fig = classifier.fourier_coefficents()
# fig.savefig("./graphs/classifier/data_reupload/sample_fourier_coeffs.svg", format='svg', bbox_inches="tight")


# classifier.train_test_split()
# classifier.learn_model(epochs=100)
# classifier.score_model()
# fig = classifier.plot_fit()
# fig.savefig("./graphs/classifier/data_reupload/sample.svg", format='svg', bbox_inches="tight")


### CONVOLUTIONAL CLASSIFICATION EXAMPLE ###

import models
from autoencoder import Autoencoder
from quanvolutional import QCNN
from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

# Load data
data, target = make_moons(n_samples=N_SAMPLES,noise=NOISE)
scalerOHE = OneHotEncoder(sparse_output=False)
encoded_target = scalerOHE.fit_transform(target.reshape(-1,1))
sScaler = StandardScaler()
encoded_data = sScaler.fit_transform(data)

# Encode the data
# autoencoder = Autoencoder(encoded_data, N_FEATURES)
# encoded_data = autoencoder.encoder(torch.from_numpy(encoded_data).float()).detach().numpy()

model = models.simple_ansatz
    
qcnn = QCNN(model, encoded_data, encoded_target, N_LAYERS)

fig = qcnn.fourier_coefficents()
fig.savefig("./graphs/convolution/classical_activation/simple_ansatz/sample_fourier_coeffs.svg", format='svg', bbox_inches="tight")

qcnn = QCNN(model, encoded_data, encoded_target, N_LAYERS)
qcnn.train_test_split()
qcnn.learn_model(epochs=100)
qcnn.score_model()
fig = qcnn.plot_fit()
fig.savefig("./graphs/convolution/classical_activation/simple_ansatz/sample.svg", format='svg', bbox_inches="tight")

