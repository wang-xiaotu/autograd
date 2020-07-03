from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import autograd.numpy as np
import autograd.scipy.linalg as spl
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
        mean, log_std, rho = params[:D], params[D:-1], params[-1:]
        print("before",rho)
        if rho >= 1 or rho <0:
             rho = np.exp(rho)/(1+np.exp(rho))
        # elif rho < 0:
        #     rho = abs(rho)
        print("after", rho)
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
        covariance = np.diag(np.exp(2 * log_std)) * (1-rho) + variance_matrix
        print(type(covariance))
        print("Det of the covariance", np.linalg.det(covariance))
        cov_decomp = np.linalg.cholesky(covariance)
        pen_rho = 0 if abs(rho)<=1 else 1e4
        samples = np.matmul(rs.randn(num_samples, len(mean)), cov_decomp) + mean
        # lower_bound = np.mean(np.log(mvn.pdf(samples, mean, covariance))) - np.mean(logprob(samples, t))
        # lower_bound = np.mean((mvn.pdf(samples, mean, covariance))) - np.mean(logprob(samples, t))
        lower_bound = mvn.entropy(mean, covariance) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def black_box_variational_inference_blockcov(logprob, D, num_samples, weight_positions):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std, rho = params[:D], params[D:-3], params[-3:]
        print("original:", rho)
        rho_p = [rho[i] if 0<rho[i]<1 else np.exp(rho[i])/(1+np.exp(rho[i])) for i in range(3)]
        rho = np.array(rho_p)
        print("after:", rho)
        # if len(rho[abs(rho)>1]) > 0:
        #     rho = np.sign(rho) * abs(rho)/(abs(rho) + 0.01)
        # for i in range(len(rho)):
        #      if rho[i] >= 1:
        #          rho = rho / (rho + 0.01)
        #      elif rho[i] <= -1:
        #          rho[i] = -0.99
        # rho[rho>=1] = 0.99
        # rho[rho<=-1] = -0.99
        # mean, log_std = params[:D], params[D:]
        # mean, rho = params[:-1], params[-1]
        return mean, log_std, rho

    rs = npr.RandomState(0)

    def get_covariance(params):
        # mat_exp = np.vectorize(my_exp)
        mean, log_std, rho = unpack_params(params)

        # mean, log_std = unpack_params(params)
        # mean, rho = unpack_params(params)

        pseudo_weight = np.ones(len(mean)).reshape(1, len(mean))
        params_positions = weight_positions(pseudo_weight)
        sigma = np.exp(log_std).reshape(len(log_std))
        variance_matrix0 = np.matmul(sigma.reshape(len(log_std), 1), sigma.reshape(len(log_std), 1).T)
        variance_matrix = np.ones(len(mean)).reshape(1, len(mean))

        start_idx = 0
        block_dim_cum = 0
        # rho = np.array((0.1,0.2,0.3))
        for k in range(len(params_positions)):
            rho_k = rho[k]
            # rho_k = 0.0001
            positions_vector = params_positions[k]
            block_dim = len(positions_vector)

            end_idx = start_idx + block_dim
            # block_matrix1 = np.diag(sigma.reshape(len(log_std))[start_idx:end_idx])
            tmp_matrix = np.array(range(block_dim)).reshape(1, block_dim)
            for row in range(1, block_dim):
                zero_elements = np.zeros(row)
                con_row = np.array(range(block_dim - row))
                complete_row = np.concatenate([zero_elements, con_row]).reshape(1, block_dim)
                tmp_matrix = np.concatenate([tmp_matrix, complete_row])

            block_matrix2 = tmp_matrix + tmp_matrix.T
            # print(block_matrix2)
            block_matrix3 = np.power(rho_k, block_matrix2)

            block_matrix = block_matrix3
            zero_matrix1 = np.zeros((block_dim, block_dim_cum))
            zero_matrix2 = np.zeros((block_dim, len(mean)-block_dim - block_dim_cum))
            part_variance = np.concatenate([zero_matrix1, block_matrix, zero_matrix2], axis=1)
            start_idx = end_idx
            # arraybox type kid of conflict with blockdiag, off diagonal won't be arraybox value
            # this cause problems in cholesky decompostion
            #variance_matrix = block_diag(variance_matrix, block_matrix)
            variance_matrix = np.concatenate([variance_matrix, part_variance])
            block_dim_cum += block_dim
        variance_matrix = np.array(variance_matrix[1:, ])
        # print(variance_matrix[0,0])

        # covariance = variance_matrix + np.diag(sigma) - np.diag(np.ones(len(mean))) #+ np.ones((len(mean), len(mean))) * 1e-10
        covariance = np.multiply(variance_matrix, variance_matrix0)
        return covariance

    def variational_objective(params, t):
        # in this covariance matrix, the correlation rho is set to the same
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std, rho = unpack_params(params)

        # mean, log_std = unpack_params(params)
        # mean, rho = unpack_params(params)
        covariance = get_covariance(params)

        cov_decomp = np.linalg.cholesky(covariance)
        # print("PDT: objective")
        # pen_rho = 0 if all(abs(rho) <= 1) else 1e10
        samples = np.matmul(rs.randn(num_samples, len(mean)), cov_decomp) + mean
        # samples = rs.multivariate_normal(mean, covariance, len(mean))
        lower_bound = mvn.entropy(mean, covariance) - np.mean(logprob(samples, t))
        return lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params, get_covariance
