from functools import partial
import pennylane.numpy as np
import pennylane as qml
import jax.numpy as jnp
import jax
import optax
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from models import qnn_compiler
from fourier import FourierCoefficents

# Class for classification tasks
class ClassifierQNN():

    def __init__(self, model: any, data: np.ndarray, target: np.ndarray, n_layers: int) -> None:
        self.target_length = target.shape[1]
        self.n_features = data.shape[1]
        
        self.model = model
        self.data = data
        self.target = target

        compiler = qnn_compiler(model, self.n_features, n_layers, self.target_length)
        self.parameter_shape = compiler.parameter_shape
        qnn = compiler.classification()
        qnn_batched = jax.vmap(qnn, (0, None))
        self.qnn = jax.jit(qnn_batched)
        
        self.fourier = FourierCoefficents(model, data, n_layers)


    def train_test_split(self, test_size:float=0.20):
        res = train_test_split(self.data, self.target, test_size=0.20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = res
        return res


    @partial(jax.jit, static_argnums=(0,))
    def cross_entropy_loss(self, y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))

    @partial(jax.jit, static_argnums=(0,))
    def calculate_ce_cost(self, X, y, theta):

        # Get the target prediction
        yp = self.qnn(X, theta)

        # Softmax the output
        yp = jax.nn.softmax(yp)

        # Compute loss metric between prediction and real
        cost = self.cross_entropy_loss(y, yp)
        return cost

    @partial(jax.jit, static_argnums=(0,))
    def optimizer_update(self, opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: self.calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


    def learn_model(self, epochs: int, seed: int = random.randrange(1000)):

        self.optimizer = optax.adam(learning_rate=0.04)

        # Seed
        key = jax.random.PRNGKey(seed)

        # Initialize circuit params
        initial_params = jax.random.normal(key, shape=self.parameter_shape)
        params = jnp.copy(initial_params)

        # Initialize optimizer
        opt_state = self.optimizer.init(initial_params)

        ##### FIT #####
        for epoch in range(epochs):
            params, opt_state, cost = self.optimizer_update(opt_state, params, self.X_train, self.y_train)
            if epoch % 5 == 0:
                print(f'epoch: {epoch}\t cost: {cost}')

            if epoch % 10 == 0:
                pass
                

        # Store trained parameters
        self.fit_params = params


    def score_model(self):

        # Evaluate the cross entropy loss on the training set
        yp = self.qnn(self.X_train, self.fit_params)
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on training set: {self.cross_entropy_loss(self.y_train,yp)}')

        # Evaluate the accuracy on the training set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on training set: {accuracy_score(jnp.argmax(self.y_train,axis=1),yp)}\n')

        # Evaluate the Cross Entropy Loss on the testing set
        yp = self.qnn(self.X_test, self.fit_params)
        yp = jax.nn.softmax(yp)
        print(f'\nCross entropy loss on test set: {self.cross_entropy_loss(self.y_test,yp)}')

        # Evaluate the accuracy on the testing set
        yp = jnp.argmax(yp, axis=1)
        print(f'Accuracy of fullmodel on test set: {accuracy_score(jnp.argmax(self.y_test,axis=1),yp)}\n')


    def plot_fit(self):

        figure = plt.figure()
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(111)

        # Apply final params to test set
        yp = self.qnn(self.X_test, self.fit_params)
        yp = jax.nn.softmax(yp)
        yp = jnp.argmax(yp, axis=1)

        # Plot the testing points
        ax.scatter(
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=yp,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # Apply final params to test set
        yp = self.qnn(self.X_train, self.fit_params)
        yp = jax.nn.softmax(yp)
        yp = jnp.argmax(yp, axis=1)

        # Plot the testing points
        ax.scatter(
            self.X_train[:, 0],
            self.X_train[:, 1],
            c=yp,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )