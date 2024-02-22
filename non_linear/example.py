import models
from autoencoder import Autoencoder
from classifier import ClassifierQNN

from sklearn.datasets import make_moons
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
autoencoder = Autoencoder(encoded_data, 5)
encoded_data = autoencoder.encoder(torch.from_numpy(encoded_data).float()).detach().numpy()

# Choose model
model = lambda n_features, n_layers: models.mid_measure(n_features, n_layers, 2)

# Train the model
classifier = ClassifierQNN(model, encoded_data, encoded_target, 4)
classifier.train_test_split()
classifier.learn_model(epochs=100)
classifier.score_model()
classifier.plot_fit()