import torch
import random
import pennylane.numpy as np
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, data:np.ndarray, bneck_size:int):

        self.bneck_size = bneck_size
        input_size = data.shape[1]

        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,32),
            nn.ReLU(),
            nn.Linear(32, bneck_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bneck_size, 32),
            nn.ReLU(),
            nn.Linear(32,input_size),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

    def train_encoder(self, data, seed:int, lr:float, epochs:int = 700):
    
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Train the model
        batch_size = 16
        for epoch in range(self.epochs):
            loss_tot=0
            for i in range(0, len(data), batch_size):
                # Get the batch
                batch_X = torch.FloatTensor(data[i:i+batch_size])
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                outputs = self(batch_X)
                loss =  criterion(outputs, batch_X)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                loss_tot += loss.item()*(len(batch_X))

            loss_tot=loss_tot/len(data)
            # Print the loss every 10 epochs
            if (epoch+1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.epochs, loss_tot))

