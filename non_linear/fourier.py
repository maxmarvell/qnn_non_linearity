from pennylane import numpy as np
import matplotlib.pyplot as plt
import jax

from numpy import ndarray

from models import qnn_compiler

# Defining a class that computes fourier coefficents given a quantum model

class SampleFourierCoefficients():
    def __init__(self, model:any, n_features:int , n_layers:int) -> None:
        
        compiler = qnn_compiler(model, n_features, n_layers, 1)
        qnn = compiler.classification()
        qnn_batched = jax.vmap(qnn, (0, None))
        
        self.parameter_shape = compiler.parameter_shape
        self.n_features = n_features
        self.qnn = jax.jit(qnn_batched)

    def get_coeffs(self,f,K):

        # To obtain K coeffs we require 2K-1 inputs
        n_coeffs = 2*K-1

        # Sample evenly spaced coefficents
        t = np.tile(np.linspace(0,2*np.pi,n_coeffs,endpoint=False), (self.parameter_shape[-1], 1)).T


        # Apply fourier transform to estimate fourier coefficents
        y = np.fft.rfft(f(t)) / t[0].size

        return y


    def random_sample(self,n_coeffs,n_samples):

        def f(x):
            # Return the parametrised circuit
            return self.qnn(x, self.params)

        self.params = np.array([jax.random.uniform(jax.random.PRNGKey(i), shape=self.parameter_shape)*2*np.pi for i in range(n_samples)])

        coeffs = self.get_coeffs(f, n_coeffs)

        self.coeffs_real = np.real(coeffs)
        self.coeffs_imag = np.imag(coeffs)

        return np.real(coeffs), np.imag(coeffs)


    def plot_coeffs(self):

        n_coeffs = len(self.coeffs_real[0])
        fig, ax = plt.subplots(1, n_coeffs, figsize=(15,4))

        for idx, ax_ in enumerate(ax):
            ax_.set_title(r"$c_{}$".format(idx))
            ax_.scatter(self.coeffs_real[:, idx], self.coeffs_imag[:, idx], s=35, facecolor='white', edgecolor='red')
            ax_.set_aspect("equal")
            ax_.set_ylim(-1, 1)
            ax_.set_xlim(-1, 1)

        plt.tight_layout(pad=0.5)
        plt.show()
        
        
        
class FourierCoefficents():
    def __init__(self, model:any, data:ndarray, n_layers) -> None:
        
        self.n_features = data.shape[1]
        self.data = data
        
        compiler = qnn_compiler(model, self.n_features, n_layers, 1)
        self.qnn = compiler.classification()
        # qnn_batched = jax.vmap(qnn, (0, None))
        
        # self.qnn = jax.jit(qnn_batched)

    def get_coeffs(self, params:ndarray):
        # Apply fourier transform to estimate fourier coefficents
        data = self.data[0]
        coeffs = np.fft.rfft(self.qnn(self.data[0], params)) / self.data[0].size
        print(self.data[0].shape)
        print(coeffs.shape)
        
        self.coeffs_real = np.real(coeffs)
        self.coeffs_imag = np.imag(coeffs)
        
        return np.real(coeffs), np.imag(coeffs)

    def plot_coeffs(self):

        n_coeffs = len(self.coeffs_real[0])
        fig, ax = plt.subplots(1, n_coeffs, figsize=(15,4))

        for idx, ax_ in enumerate(ax):
            ax_.set_title(r"$c_{}$".format(idx))
            ax_.scatter(self.coeffs_real[:, idx], self.coeffs_imag[:, idx], s=35, facecolor='white', edgecolor='red')
            ax_.set_aspect("equal")
            ax_.set_ylim(-1, 1)
            ax_.set_xlim(-1, 1)

        plt.tight_layout(pad=0.5)
        plt.show()
