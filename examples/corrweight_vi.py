from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


def black_box_variational_inference_corsigma(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std, rho = params[:D], params[D:-1], params[-1]
        return mean, log_std, rho

    def gaussian_entropy(log_std, rho):
        sigma = np.exp(log_std).reshape(len(log_std), 1)
        variance_matrix = np.matmul(sigma, sigma.T) * rho
        covariance = np.diag(np.exp(2 * log_std)) * (1-rho) + variance_matrix \
                     + np.diag(1e-10 * np.ones(len(log_std)))
        return 0.5 * D * (1 + np.log(2*np.pi)) + 0.5 * np.linalg.det(covariance)
        # return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        # in this covariance matrix, the correlation rho is set to the same
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, rho = unpack_params(params)
        sigma = np.exp(log_std).reshape(len(log_std), 1)
        variance_matrix = np.matmul(sigma, sigma.T) * rho
        covariance = np.diag(np.exp(2 * log_std)) * (1-rho) + variance_matrix \
                     + np.diag(1e-10 * np.ones(len(mean)))
        print(np.linalg.det(covariance))
        cov_decomp = np.linalg.cholesky(covariance)
        samples = np.matmul(rs.randn(num_samples, len(mean)), cov_decomp) + mean
        # lower_bound = np.mean(np.log(mvn.pdf(samples, mean, covariance))) - np.mean(logprob(samples, t))
        lower_bound = np.mean((mvn.pdf(samples, mean, covariance))) - np.mean(logprob(samples, t))
        # lower_bound = gaussian_entropy(log_std, rho) + np.mean(logprob(samples, t))
        return lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def black_box_variational_inference_blockcov(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std, rho = params[:D], params[D:-1], params[-1]
        return mean, log_std, rho

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        # in this covariance matrix, the correlation rho is set to the same
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, rho = unpack_params(params)
        sigma = np.exp(log_std).reshape(len(log_std), 1)
        variance_matrix = np.matmul(sigma, sigma.T) * rho
        covariance = np.diag(np.exp(2 * log_std)) * (1-rho) + variance_matrix \
                     + np.diag(1e-10 * np.ones(len(mean)))
        print(np.linalg.det(covariance))
        cov_decomp = np.linalg.cholesky(covariance)
        samples = np.matmul(rs.randn(num_samples, len(mean)), cov_decomp) + mean
        lower_bound = np.mean(mvn.pdf(samples, mean, covariance)) - np.mean(logprob(samples, t))
        return lower_bound



    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params