from pennylane import numpy as np
import matplotlib.pyplot as plt
import jax

from numpy import ndarray

from non_linear.models import qnn_compiler

# Defining a class that computes fourier coefficents given a quantum model

class SampleFourierCoefficients():
    def __init__(self, model:any, parameter_shape:tuple, n_features:int) -> None:
        self.model = model
        self.parameter_shape = parameter_shape
        self.n_features = n_features

    def get_coeffs(self,f,K):

        # To obtain K coeffs we require 2K-1 inputs
        n_coeffs = 2*K-1

        # Sample evenly spaced coefficents
        t = np.tile(np.linspace(0,2*np.pi,n_coeffs,endpoint=False), (self.n_features, 1)).T

        # Apply fourier transform to estimate fourier coefficents
        y = np.fft.rfft(f(t).reshape(-1,n_coeffs)) / t[0].size

        return y


    def random_sample(self,n_coeffs,n_samples):

        def f(x):
            # Return the parametrised circuit
            batched_model = jax.vmap(self.model, (None, 0))
            model = jax.jit(batched_model)
            return model(x, self.params)

        self.params = np.array([jax.random.uniform(jax.random.PRNGKey(i), shape=self.parameter_shape)*2*np.pi for i in range(n_samples)])

        coeffs = self.get_coeffs(f, n_coeffs)

        self.coeffs_real = np.real(coeffs)
        self.coeffs_imag = np.imag(coeffs)

        return np.real(coeffs), np.imag(coeffs)


    def plot_coeffs(self, show:bool = False, ax = None):

        # get number of fourier coeffs
        n_coeffs = len(self.coeffs_real[0])
        
        # if no array of axes provided set a default
        if type(ax) != ndarray: 
            fig, ax = plt.subplots(1, n_coeffs, figsize=(15,4))
            plt.tight_layout(pad=0.5)

        assert len(ax) == n_coeffs, "Length of axes should be same as number of fourier coefficents!"

        for idx, ax_ in enumerate(ax):
            ax_.set_title(r"$c_{}$".format(idx))
            ax_.scatter(self.coeffs_real[:, idx], self.coeffs_imag[:, idx], s=35, facecolor='white', edgecolor='red')
            ax_.set_aspect("equal")
            ax_.set_ylim(-1, 1)
            ax_.set_xlim(-1, 1)

        if show: plt.show()

        if type(ax) != ndarray: return fig
