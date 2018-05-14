from math import sqrt

import autograd.numpy as np
import autograd.numpy.random as npr

from acos_dist import acosdist
from utils import kl_div, l2_norm


def initialize_parameters(layer_sizes, xavier=False, var=0.1):
  nb_layers = len(layer_sizes) - 1
  parameters = []
  for i in range(nb_layers):
    matrix_size = layer_sizes[i] * layer_sizes[i + 1]

    if xavier:
      var = sqrt(2. / (layer_sizes[i] + layer_sizes[i + 1]))
    else:
      var = var

    W = npr.normal(0, var, matrix_size).reshape(
        (layer_sizes[i], layer_sizes[i + 1]))
    b = np.zeros(layer_sizes[i + 1])
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


def error(parameters, X_train, y_train, activations, y_type, loss_type):
  y_pred = fully_connected(X_train, parameters, activations)
  if y_type == "coord":
    return acosdist(y_pred, y_train)
  if y_type == "heatmap":
    if loss_type == "kl":
      return kl_div(y_pred, y_train)
    if loss_type == "emd":
      pass


def reg_loss(parameters, L2_reg):
  loss = 0
  for W, b in parameters:
    loss += l2_norm(W) + l2_norm(b)
  return L2_reg * loss


def loss(parameters, X_train, y_train, L2_reg, activations, y_type,
    loss_type=None):
  base_error = error(parameters, X_train, y_train, activations, y_type,
                     loss_type)
  reg_error = reg_loss(parameters, L2_reg)
  return base_error + reg_error
