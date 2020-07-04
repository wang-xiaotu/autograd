from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import csv
import sys
import numpy
import autograd.numpy as np
import autograd.numpy.random as npr
from black_box_svi import black_box_variational_inference
from autograd.misc.optimizers import adam
import random

def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m + 1) * n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            # weights
            yield weights[:, :m * n].reshape((num_weight_sets, m, n)), \
                  weights[:, m * n:m * n + n].reshape((num_weight_sets, 1, n)) # constant unit
            weights = weights[:, (m + 1) * n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    def truepost(weights, inputs, targets):
        log_prior = -np.sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


# def build_toy_dataset(n_data=40, noise_std=0.1, seed = 0):
    # D = 1
    # rs = npr.RandomState(seed)
    # total_num = n_data * 10
    # overallinput = np.linspace(-8, 8, num=total_num)
    # overalltarget = np.cos(overallinput) + rs.randn(n_data * 10) * noise_std
    # overallinput = overallinput / 4
    # input_ix1 = np.logical_and(overallinput < 1.2, overallinput > 0.4)
    # input_ix2 = np.logical_and(overallinput < -0.2, overallinput > -0.6)
    # input_ix = np.logical_or(input_ix1, input_ix2)
    # inputs = overallinput[input_ix]
    # targets = overalltarget[input_ix]
    # inputs = inputs.reshape((len(inputs), D))
    # targets = targets.reshape((len(targets), D))
    # overallinput = overallinput.reshape((len(overallinput), D))
    # overalltarget = overalltarget.reshape((len(overallinput), D))
    #
    # return inputs, targets, overallinput, overalltarget, input_ix

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
    input_args = sys.argv
    print(sys.argv)
    B = int(sys.argv[1])
    t = sys.argv[2]
    tau = float(sys.argv[3])
    inputs, targets, tot_inputs, tot_targets, input_idx = build_toy_dataset(n_data=40, noise_std=tau, type = t)
    coverage_df = np.zeros(400).reshape(400, 1)
    coverage_df[input_idx, :] = 1 # indicate the training data index
    for b in range(B):
        # Specify inference problem by its unnormalized log-posterior.
        rbf = lambda x: np.exp(-x ** 2)
        num_weights, predictions, logprob = \
            make_nn_funs(layer_sizes=[1, 20, 20, 1], L2_reg=0.1,
                         noise_variance=0.01, nonlinearity=rbf)

        inputs, targets, tot_inputs, tot_targets, input_idx = build_toy_dataset(n_data=40, noise_std=tau, type = t)
        log_posterior = lambda weights, t: logprob(weights, inputs, targets)

        # Build variational objective.
        objective, gradient, unpack_params = \
            black_box_variational_inference(log_posterior, num_weights,
                                            num_samples=20)

    # Set up figure.
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)


        def callback(params, t, g):
            print("Iteration {} lower bound {}".format(t, -objective(params, t)))

            # Sample functions from posterior.
            rs = npr.RandomState(0)
            mean, log_std = unpack_params(params)
            # rs = npr.RandomState(0)
            sample_weights = rs.randn(10, num_weights) * np.exp(log_std) + mean
            plot_inputs = np.linspace(-2, 2, num=400)
            outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

            # Plot data and functions.
            plt.cla()
            ax.plot(tot_inputs.ravel(), tot_targets.ravel(), "bx", label="testing data")
            ax.plot(inputs.ravel(), targets.ravel(), 'rx', label="training data")
            ax.plot(plot_inputs, outputs[:, :, 0].T)
            ax.set_ylim([-2, 3])
            ax.legend()
            plt.draw()
            plt.pause(1.0 / 60.0)


        # Initialize variational parameters
        rs = npr.RandomState(0)
        init_mean = rs.randn(num_weights)
        init_log_std = -5 * np.ones(num_weights)
        init_var_params = np.concatenate([init_mean, init_log_std])

        print("Optimizing variational parameters...")
        variational_params = adam(gradient, init_var_params,
                                  step_size=0.1, num_iters=500, callback=callback)
        # print(variational_params)
        #
        # Sample functions from the final posterior.
        rs = npr.RandomState(0)
        mean, log_std = unpack_params(variational_params)
        # rs = npr.RandomState(0)
        sample_weights = rs.randn(1000, num_weights) * np.exp(log_std) + mean
        plot_inputs = np.linspace(-2, 2, num=400)
        outputs_final = predictions(sample_weights, np.expand_dims(plot_inputs, 1))
        lowerbd = numpy.quantile(outputs_final, 0.05, axis=0)
        upperbd = numpy.quantile(outputs_final, 0.95, axis=0)
        inconint = np.logical_and(lowerbd < tot_targets, upperbd > tot_targets).ravel()
        con_ind = np.zeros(len(lowerbd))
        con_ind[inconint] = 1
        con_ind = con_ind.reshape(len(con_ind), 1)
        coverage_df = np.concatenate([coverage_df, con_ind], axis=1)
        # # Plot data and functions.
        # fig = plt.figure(figsize=(12, 8), facecolor='white')
        # ax = fig.add_subplot(111, frameon=False)
        # ax.plot(tot_inputs.ravel(), tot_targets.ravel(), "bx", label="testing data")
        # ax.plot(inputs.ravel(), targets.ravel(), 'rx', label="training data")
        # ax.plot(tot_inputs.ravel(), lowerbd.ravel(), "k-", label="error bar")
        # ax.plot(tot_inputs.ravel(), upperbd.ravel(), "k-")
        # ax.legend()
        # ax.set_ylim([tot_targets.min()-0.1,tot_targets.max()+0.1])
        # plt.show()

    filename = "diagVar" + "B" + str(B) + "t"+str(t) + "noise"+str(tau) + ".csv"

    np.savetxt(filename, coverage_df, delimiter=',', fmt='%d')

    # csv.reader("diagVar_QuantUQ.csv")