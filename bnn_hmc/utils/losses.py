# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Likelihood and prior functions."""
import math

import jax
import jax.numpy as jnp

from bnn_hmc.utils import tree_utils


def make_xent_log_likelihood(temperature):
  def xent_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood."""
    _, y = batch
    logits, net_state = net_apply(params, net_state, None, batch, is_training)
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(y, num_classes)
    softmax_xent = jnp.sum(labels * jax.nn.log_softmax(logits)) / temperature

    return softmax_xent, net_state
  return xent_log_likelihood


def preprocess_network_outputs_gaussian(predictions):
  """Apply softplus to std output if available.

  Returns predictive mean and standard deviation.
  """
  predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
  predictions_std = jax.nn.softplus(predictions_std)
  return jnp.concatenate([predictions_mean, predictions_std], axis=-1)


def make_gaussian_likelihood(temperature):
  def gaussian_log_likelihood(net_apply, params, net_state, batch, is_training):
    """Computes the negative log-likelihood.

    The outputs of the network should be two-dimensional.
    The first output is treated as predictive mean. The second output is treated
    as inverse-softplus of the predictive standard deviation.
    """
    _, y = batch
    predictions, net_state = net_apply(
        params, net_state, None, batch, is_training)

    predictions = preprocess_network_outputs_gaussian(predictions)
    predictions_mean, predictions_std = jnp.split(predictions, [1], axis=-1)
    tempered_std = predictions_std * jnp.sqrt(temperature)
      
    se = (predictions_mean - y)**2
    log_likelihood = (-0.5 * se / tempered_std**2
                      - 0.5 * jnp.log(tempered_std**2 * 2 * math.pi))
    log_likelihood = jnp.sum(log_likelihood)
    
    return log_likelihood, net_state
  return gaussian_log_likelihood


def make_gaussian_log_prior(weight_decay, temperature):
  """Returns the Gaussian log-density and delta given weight decay."""

  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    # ToDo izmailovpavel: make temperature treatment the same as in gaussian
    # likelihood function.
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    log_prob = -(0.5 * tree_utils.tree_dot(params, params) * weight_decay +
                 0.5 * n_params * jnp.log(weight_decay / (2 * math.pi)))
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior log-density."""
    diff = sum([
        jnp.sum(p1**2 - p2**2)
        for p1, p2 in zip(jax.tree_leaves(params1), jax.tree_leaves(params2))
    ])
    return -0.5 * weight_decay * diff / temperature

  return log_prior, log_prior_diff


def make_laplace_log_prior(weight_decay, temperature):
  """Returns the Gaussian log-density and delta given weight decay."""

  b = jnp.sqrt(1 / 2 / weight_decay) # 2 * b**2 == 1/ weight_decay

  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    l1_norm = sum([jnp.abs(p).sum() for p in jax.tree_leaves(params)])
    log_prob = -l1_norm / b - jnp.log(2 * b)
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior log-density."""
    diff = sum([
        (jnp.abs(p1) - jnp.abs(p2)).sum()
        for p1, p2 in zip(jax.tree_leaves(params1), jax.tree_leaves(params2))
    ])
    return  -diff / b / temperature

  return log_prior, log_prior_diff


def make_student_t_log_prior(weight_decay, degrees_of_freedom, temperature):
  def log_prior(params):
    log_prob = sum([jnp.log(1 + p**2 * weight_decay / degrees_of_freedom).sum()
                    for p in jax.tree_leaves(params)])
    log_prob *= -(degrees_of_freedom + 1) / 2
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    return log_prior(params1) - log_prior(params2)

  return log_prior, log_prior_diff


def make_exp_fnorm_p_log_prior(weight_decay, power, temperature):
  """Returns the exp(- F-norm^p) prior log-density and delta given weight decay.

  Gaussian prior corresponds to power p=2.
  """

  def log_prior(params):
    """Computes the Gaussian prior log-density up to a constant."""
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    fnorm_p = (tree_utils.tree_dot(params, params))**(power / 2)
    log_prob = -0.5 * fnorm_p * weight_decay
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    """Computes the delta in  Gaussian prior log-density."""
    return log_prior(params1) - log_prior(params2)

  return log_prior, log_prior_diff


def make_lenet_conv_correlated_log_prior(weight_decay, inv_sigma, temperature):
  """Returns the block-correlated log-density and delta given weight decay."""

  gaussian_prior, gaussian_diff = make_gaussian_log_prior(
      weight_decay, temperature)
  def log_prior(params):
    """Computes the Gaussian prior log-density."""
    n_params = sum([p.size for p in jax.tree_leaves(params)])
    log_prob = - 0.5 * jnp.trace(jnp.matmul(
      jnp.matmul(params["conv2_d"]["w"].reshape(25, 6).T, inv_sigma),
      params["conv2_d"]["w"].reshape(25, 6)))
    log_prob += gaussian_prior(params)
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    """Computes the delta in block-correlated prior log-density."""
    return (log_prior(params1) - log_prior(params2) +
            gaussian_diff(params1, params2))

  return log_prior, log_prior_diff


def make_lenet_sumfilter_log_prior(
  weight_decay, filter_weight_decay, temperature
):
  gaussian_prior, gaussian_diff = make_gaussian_log_prior(
    weight_decay, temperature)
  def log_prior(params):
    kw, kh, cin, cout = params["conv2_d"]["w"].shape
    sum_filter = jnp.sum(jnp.abs(
      jnp.sum(params["conv2_d"]["w"].reshape(kw * kh, cin, cout), 0)))
    log_prob = -0.5 * sum_filter * weight_decay
    log_prob += gaussian_prior(params)
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    return log_prior(params1) - log_prior(params2)

  return log_prior, log_prior_diff


def make_lenet_pca_log_prior(
  weight_decay, pca_inv_cov, pca_wd, temperature
):
  gaussian_prior, gaussian_diff = make_gaussian_log_prior(
    weight_decay, temperature)
  def log_prior(params):
    kw, kh, cin, cout = params["conv2_d"]["w"].shape
    w = params["conv2_d"]["w"].reshape(kw * kh * cin, cout)
    log_prob = - jnp.einsum("ic,ij,jc->", w, pca_inv_cov, w)
    log_prob *= pca_wd / 2

    log_prob += gaussian_prior(params)
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    return log_prior(params1) - log_prior(params2)

  return log_prior, log_prior_diff


def make_mlp_pca_log_prior(
  weight_decay, pca_inv_cov, pca_wd, temperature
):
  gaussian_prior, gaussian_diff = make_gaussian_log_prior(
    weight_decay, temperature)
  def log_prior(params):
    # n_in, n_out = params["linear"]["w"].shape
    w = params["linear"]["w"]
    log_prob = - jnp.einsum("ic,ij,jc->", w, pca_inv_cov, w)
    log_prob *= pca_wd / 2

    log_prob += gaussian_prior(params)
    return log_prob / temperature

  def log_prior_diff(params1, params2):
    return log_prior(params1) - log_prior(params2)

  return log_prior, log_prior_diff