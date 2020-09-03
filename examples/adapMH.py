from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import random
import pyreadr
import sys
import numpy
import torch
import torch.optim as optim
import copy
from autograd import grad
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
        log_prior = -0.5 * np.sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -0.5 * np.sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob, weights_position


def build_toy_dataset(n_data=40, noise_std=0.1, type="1", seed=0):
    D = 1
    rs = npr.RandomState(seed)
    total_num = n_data * 10
    overallinput = np.linspace(-8, 8, num=total_num)
    overalltarget = 0.7 * np.cos(overallinput) + 0.25 * overallinput + rs.randn(n_data * 10) * noise_std
    overallinput = overallinput / 4
    if type == "1":
        # print("more")
        input_ix = random.sample(range(total_num), 200)
    elif type == "0":
        # print("less")
        # input_ix = np.concatenate([random.sample()])
        input_ix1 = np.array(range(total_num))[overallinput <= 1]
        input_ix11 = random.sample(input_ix1.tolist(), 190)
        input_ix2 = np.array(range(total_num))[overallinput > 1]
        input_ix22 = random.sample(input_ix2.tolist(), 10)

        # input_ix1 = np.logical_and(overallinput <=1, overallinput > 0.4)
        # input_ix2 = np.logical_and(overallinput <= 0.4, overallinput > -1)
        input_ix = np.concatenate([input_ix11, input_ix22])

    inputs = overallinput[input_ix]
    targets = overalltarget[input_ix]
    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    overallinput = overallinput.reshape((len(overallinput), D))
    overalltarget = overalltarget.reshape((len(overallinput), D))

    return inputs, targets, overallinput, overalltarget, input_ix


if __name__ == '__main__':
    B = 100000
    rbf = lambda x: np.exp(-x ** 2)
    num_weights, predictions, logprob, w_positions = \
        make_nn_funs(layer_sizes=[1, 2, 2, 1], L2_reg=0.1,
                     noise_variance=0.01, nonlinearity=rbf)
    inputs, targets, tot_inputs, tot_targets, input_idx = build_toy_dataset(n_data=40)
    beta = 0.98
    tmp = pyreadr.read_r('hmcMean')
    init = tmp['hmcsample_mean'].to_numpy()

    #Let the proposal distribution being the q(xp|x) ~ (1-beta)N(x, 2.38^2Sigma/d) + betaN(x, (0.01)^2I/d)
    # for k in range(1):
    #     rs = npr.RandomState(k)
    #     ws = init
    #     # ws = rs.randn(num_weights)
    #     ws_df = ws.reshape(num_weights, 1)
    #     # accp_df = np.zeros(num_weights).reshape(num_weights, 1)
    #     accp = np.zeros(B)
    #     # logs = np.zeros(num_weights)
    #     # logs_df = logs.reshape(num_weights, 1)
    #
    #     for b in range(B):
    #         print(b)
    #
    #         if b == 0:
    #             Sigma = np.diag(np.ones(num_weights))
    #         else:
    #             mu_hat = np.mean(ws_df, axis=1)
    #             mu = np.repeat(mu_hat, (b + 1)).reshape(ws_df.shape)
    #             tmp = ws_df - mu
    #             Sigma10 = np.matmul(tmp, tmp.T) / (b + 1)
    #             Sigma1 = (2.38 * (1 - beta)) ** 2 * Sigma10 / (num_weights)
    #             Sigma2 = (0.01 * beta) ** 2 * np.diag(np.ones(num_weights)) / num_weights
    #             Sigma = Sigma1 + Sigma2
    #         print("org", ws)
    #         ws_tmp = rs.multivariate_normal(ws.reshape(num_weights), Sigma, 1)
    #         print("new", ws_tmp)
    #         Pprime = (logprob(ws_tmp.reshape(1, num_weights), inputs, targets))
    #         print("Pprime", Pprime)
    #         P0 = (logprob(ws.reshape(1, num_weights), inputs, targets))
    #         print("P0", P0)
    #         alpha = np.min(((Pprime - P0), 0))
    #         print("alpha", alpha)
    #         bb = np.log(rs.rand(1))
    #         print(bb)
    #         accp[b] = 1 if bb <= alpha else 0
    #         ws = ws_tmp if bb <= alpha else ws
    #
    #         ws_df = np.concatenate([ws_df, ws.reshape(num_weights, 1)], axis=1)
    #         # logs_df = np.concatenate([logs_df, logs.reshape(num_weights, 1)], axis=1)
    #         # accp_df = np.concatenate([accp_df, accp.reshape(num_weights, 1)], axis=1)
    #         #a = b % 100000
    #         #if a == 0:
    #     filename = "adp_mh_wts2_826" + "K" + str(k)  + ".csv"
    #     np.savetxt(filename, ws_df, delimiter=',', fmt='%f')
    #
    #     accp_file = "adp_mh_accp_826" + "K" + str(k) + ".csv"
    #     # logs_file = "adp_logs_824" + "K" + str(k) + ".csv"
    #     # np.savetxt(logs_file, logs_df, delimiter=',', fmt='%f')
    #     np.savetxt(accp_file, accp, delimiter=',', fmt='%d')

    w0 = np.random.randn(1, num_weights)
    plot_inputs = np.linspace(-2, 2, num=400)
    outputs = predictions(w0, np.expand_dims(plot_inputs, 1))
    #outputs = outputs.to_numpy()
    # plt.cla()
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    ax.plot(tot_inputs.ravel(), tot_targets.ravel(),"k.")
    # ax.plot(inputs.ravel(), targets.ravel(), 'rx', label="training data")
    ax.plot(plot_inputs, o0[:, :, 0].T, "green", label = "w0")
    ax.plot(plot_inputs, o1[:, :, 0].T, "red", label = "w1")
    ax.plot(plot_inputs, o2[:, :, 0].T,"blue", label = "w2")
    ax.legend()
    ax.set_ylim([tot_targets.min() - 0.1, tot_targets.max() + 0.1])
    plt.draw()
