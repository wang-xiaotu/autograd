from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import random
import sys
import numpy
import pyreadr
import tensorflow as tf
import tensorflow_probability as tfp
import copy
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
        num_weight_sets = tf.size(weights)/num_weights
        for m, n in shapes:
            yield tf.reshape(weights[:, :m * n], [num_weight_sets, m, n]), \
                  tf.reshape(weights[:, m * n:m * n + n], [num_weight_sets, 1, n])
            weights = weights[:, (m + 1) * n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = tf.einsum('mnd,mdo->mno', inputs, W) + b
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
        log_prior = -0.5*tf.reduce_sum(weights ** 2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -0.5*tf.reduce_sum((preds - targets) ** 2, axis=1)[:, 0] / noise_variance
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
    B = 1
    sam = np.ones(13).reshape(1, 13)
    rbf = lambda x: tf.exp(-x ** 2)
    num_weights, predictions, logprob, w_positions = \
        make_nn_funs(layer_sizes=[1, 2, 2, 1], L2_reg=0.1,
                     noise_variance=0.01, nonlinearity=rbf)
    inputs, targets, tot_inputs, tot_targets, input_idx = build_toy_dataset(n_data=40)

    def target_logprob(w):
        return logprob(w, inputs, targets)


    # Initialize the HMC transition kernel.
    num_results = int(3e6)
    num_burnin_steps = int(2e4)
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_logprob,
            num_leapfrog_steps=3,
            step_size=0.9),
        num_adaptation_steps=int(num_burnin_steps * 0.8))


    # Run the chain (with burn-in).
    @tf.function
    def run_chain(start_point):
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state = start_point,
            #current_state=np.random.randn(1, num_weights),
            kernel=adaptive_hmc,

            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

        sample_mean = tf.reduce_mean(samples)
        sample_stddev = tf.math.reduce_std(samples)
        is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
        return sample_mean, sample_stddev, is_accepted, samples


    tmp = pyreadr.read_r('hmcstartpoints')
    tmp1 = tmp['D1'].to_numpy()
    tmp2 = tmp1.reshape(13, 100)

    for start in range(100):
        spts = tmp2[:,start]
        sample_mean, sample_stddev, is_accepted, samples = run_chain(spts.reshape(1,13))

        print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))

        sam = samples.numpy()
        sam = sam.reshape(num_results, 13)
        sam = np.concatenate([sam, sam])
    np.savetxt("multiple_hmc.csv", sam, delimiter=',', fmt='%f')

