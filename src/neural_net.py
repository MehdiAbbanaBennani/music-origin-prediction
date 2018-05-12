import autograd.numpy as np
import autograd.numpy.random as npr
from acos_dist import acosdist
from autograd.misc.flatten import flatten

from numpy.random import permutation


def batch_generator(X, y, batch_size):
  n_obs = X.shape[0]
  while True :
    indices = permutation(n_obs)
    curr_idx = 0
    while curr_idx < n_obs :
      next_idx = min(curr_idx + batch_size, n_obs)
      batch_indices = indices[slice(curr_idx, next_idx)]
      batch = (X[batch_indices], y[batch_indices])
      curr_idx += batch_size
      yield batch

def initialize_parameters(layer_sizes) :
  nb_layers = len(layer_sizes) - 1
  parameters = []
  for i in range(nb_layers):
    matrix_size = layer_sizes[i] * layer_sizes[i+1]
    W = npr.normal(0, 0.1, matrix_size).reshape((layer_sizes[i], layer_sizes[i+1]))
    b = np.zeros(layer_sizes[i+1])
    parameters.append((W, b))
  return parameters

def fully_connected(X, parameters, activation_functions):
  """ Returns a fully connected function"""
  nb_layers = len(activation_functions)
  input = X
  for i in range(nb_layers):
    W, b = parameters[i]
    z = np.dot(input, W) + b
    input = activation_functions[i](z)
  return input

def l2_norm(params):
  """Computes l2 norm of params by flattening them into a vector."""
  flattened, _ = flatten(params)
  return np.dot(flattened, flattened)

def regularization_loss(parameters):
  loss = 0
  for W, b in parameters :
    loss += l2_norm(W) + l2_norm(b)
  return loss

def loss(parameters, X_train, y_train, L2_reg, activations):
  y_pred = fully_connected(X_train, parameters, activations)
  return acosdist(y_pred, y_train) + L2_reg* regularization_loss(parameters)
