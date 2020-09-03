from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import random
import sys
import copy
import numpy
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.misc.optimizers import adam


def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m + 1) * n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m * n].reshape((num_weight_sets, m, n)), \
                  weights[:, m * n:m * n + n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m + 1) * n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def weights_position(weights):
        layer_ix = 0
        positions = [1, 2, 3]
        for W, b in unpack_layers(weights):
            if len(W[0]) > 1:
                len_w = len(W[0]) * len(W[0][0])
            else:
                len_w = len(W[0][0])
            len_b = len(b[0][0])
            len_current_layer = len_w + len_b
            positions[layer_ix] = range(len_current_layer) + np.array(1)
            layer_ix += 1
        return positions

    def logprob(weights, inputs, targets):
        log_prior = np.sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik
    return num_weights, predictions, logprob, weights_position


def build_toy_dataset(n_data=40, noise_std=0.1, type = "1", seed=0):
    D = 1
    rs = npr.RandomState(seed)
    total_num = n_data * 10
    overallinput = np.linspace(-8, 8, num=total_num)
    overalltarget = 0.7 * np.cos(overallinput) + 0.25*overallinput + rs.randn(n_data * 10) * noise_std
    overallinput = overallinput / 4
    if type == "1":
        #print("more")
        input_ix = random.sample(range(total_num), 200)
    elif type == "0":
        #print("less")
        #input_ix = np.concatenate([random.sample()])
        input_ix1 = np.array(range(total_num))[overallinput<=1]
        input_ix11 = random.sample(input_ix1.tolist(), 190)
        input_ix2 = np.array(range(total_num))[overallinput>1]
        input_ix22 = random.sample(input_ix2.tolist(), 10)

        #input_ix1 = np.logical_and(overallinput <=1, overallinput > 0.4)
        #input_ix2 = np.logical_and(overallinput <= 0.4, overallinput > -1)
        input_ix = np.concatenate([input_ix11, input_ix22])

    inputs = overallinput[input_ix]
    targets = overalltarget[input_ix]
    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    overallinput = overallinput.reshape((len(overallinput), D))
    overalltarget = overalltarget.reshape((len(overallinput), D))

    return inputs, targets, overallinput, overalltarget, input_ix


if __name__ == '__main__':
    B = 3000
    rbf = lambda x: np.exp(-x ** 2)
    num_weights, predictions, logprob, w_positions = \
        make_nn_funs(layer_sizes=[1, 2, 2, 1], L2_reg=0.1,
                     noise_variance=0.01, nonlinearity=rbf)
    inputs, targets, tot_inputs, tot_targets, input_idx = build_toy_dataset(n_data=40)

    # Let the proposal distribution being the q(xp|x) ~ N(xp|x, I)
    # since we would like to get the marginal distribution of each weigths
    # Metropolis within Gibbs is used.
    for k in range(1):
        rs = npr.RandomState(k)

        ws = rs.randn(num_weights)
        ws_df = np.zeros(num_weights).reshape(num_weights, 1)

        for b in range(B):
            print("b",b)
            for i in range(num_weights):
                # wiprime|wi ~ N(wi, 1)
                print("i",i)
                # Let the first 9999 to burn-in
                for m in range(50):
                    wi = ws[i]
                    wi_prime = rs.randn(1) + wi
                    ws_tmp = copy.deepcopy(ws)
                    ws_tmp[i] = wi_prime
                    Pprime = (logprob(ws_tmp.reshape(1,num_weights), inputs, targets))
                    P0 = (logprob(ws.reshape(1, num_weights), inputs, targets))
                    alpha = (Pprime - P0)
                    print("alpha", alpha)
                    bb = np.log(rs.rand(1))
                    print("b",bb)
                    wi = wi_prime if bb < alpha else wi
                    ws[i] = wi
            ws_df = np.concatenate([ws_df, ws.reshape(num_weights, 1)], axis=1)
        filename = "gibbs_wts" + "K" + str(k) + ".csv"
        np.savetxt(filename, ws_df, delimiter=',', fmt='%d')

