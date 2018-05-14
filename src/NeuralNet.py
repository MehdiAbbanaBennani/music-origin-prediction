import json

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from numpy import ceil

from Coordinates import Coordinates
from neural_net import initialize_parameters, loss, error, \
  reg_loss, fully_connected
from parameters import BATCH_SIZE, EPOCHS, PREDICT_FILE
from utils import load_data, softmax, relu


class NeuralNet:
  def __init__(self, hyperparameters):
    # middle_size, L2_reg, step_size, w_var, y_type,
    # activation, loss_type
    self.activation_map = {"relu": relu,
                           "tanh": np.tanh}
    self.layer_sizes = self.compute_layer_sizes(hyperparameters["middle_size"],
                                                hyperparameters["y_type"])
    self.L2_reg = hyperparameters["L2_reg"]
    self.activations = self.compute_activations(hyperparameters["activation"],
                                                hyperparameters["y_type"])
    self.step_size = hyperparameters["step_size"]
    self.y_type = hyperparameters["y_type"]
    self.w_var = hyperparameters["w_var"]
    self.loss_type = self.process_loss_type(hyperparameters["loss_type"])

    self.Coordinates = None
    self.optimized_params = None

  def process_loss_type(self, loss_type):
    if loss_type == "None" :
      return None
    return loss_type

  def compute_activations(self, activation, y_type):
    if y_type == "coord":
      return [self.activation_map[activation], np.tanh]
    if y_type == "heatmap":
      return [np.tanh, softmax]

  def compute_layer_sizes(self, middle_size, y_type):
    if y_type == "coord":
      return [116, middle_size, 2]
    if y_type == "heatmap":
      return [116, middle_size, 400]

  def run(self):
    L2_reg = self.L2_reg
    activations = self.activations
    step_size = self.step_size
    y_type = self.y_type
    loss_type = self.loss_type

    # Initial neural net parameters
    init_params = initialize_parameters(self.layer_sizes, var=self.w_var)

    print("Loading training data...")
    X_train, X_test, y_train, y_test = load_data(self.y_type)
    self.store(X_train, X_test, y_train, y_test)
    self.Coordinates = Coordinates(np.concatenate((y_train, y_test), axis=0))
    num_batches = int(ceil(X_train.shape[0] / BATCH_SIZE))

    def batch_indices(iter):
      if iter % num_batches == 0:
        # Shuffle the data
        X_train, X_test, y_train, y_test = load_data(self.y_type)
        self.store(X_train, X_test, y_train, y_test)
      idx = iter % num_batches
      return slice(idx * BATCH_SIZE, (idx + 1) * BATCH_SIZE)

    def objective(parameters, iter):
      idx = batch_indices(iter)
      return loss(parameters, X_train[idx], y_train[idx], L2_reg, activations,
                  y_type, loss_type)

    objective_grad = grad(objective)

    def print_perf(parameters, iter, gradient):
      if iter % num_batches == 0:
        train_acc = error(parameters, X_train, y_train, activations, y_type, loss_type)
        test_acc = error(parameters, X_test, y_test, activations, y_type, loss_type)
        reg = reg_loss(parameters, L2_reg)
        print(
            "{:15}|{:20}|{:20}|{:20}".format(iter // num_batches,
                                                   train_acc, test_acc,
                                                   reg))

    print("Training the neural network ...")
    self.optimized_params = adam(objective_grad,
                                 init_params,
                                 step_size=step_size,
                                 num_iters=EPOCHS * num_batches,
                                 callback=print_perf)
    return self.results(self.optimized_params, activations, L2_reg,
                        X_train, X_test, y_train, y_test)

  def results(self, parameters, activations, L2_reg,
      X_train, X_test, y_train, y_test):
    train_acc = error(parameters, X_train, y_train, activations, self.y_type, self.loss_type)
    test_acc = error(parameters, X_test, y_test, activations, self.y_type, self.loss_type)
    reg = reg_loss(parameters, L2_reg)
    return {"train_acc": train_acc,
            "test_acc": test_acc,
            "reg": reg,
            }

  def store(self, X_train, X_test, y_train, y_test):
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

  def predict(self):
    y_train_pred = fully_connected(self.X_train,
                                   self.optimized_params,
                                   self.activations)
    y_train_pred = self.Coordinates.min_coords(y_train_pred)

    y_test_pred = fully_connected(self.X_test,
                                  self.optimized_params,
                                  self.activations)
    y_test_pred = self.Coordinates.min_coords(y_test_pred)
    return y_train_pred, y_test_pred

  def store_predictions(self, filename=PREDICT_FILE):
    y_train_pred, y_test_pred = self.predict()
    y_train = [list(y) for y in self.y_train]
    y_test = [list(y) for y in self.y_test]

    y_dict = {"y_train_pred": y_train_pred,
              "y_test_pred": y_test_pred,
              "y_train": y_train,
              "y_test": y_test}

    with open(filename, 'w') as outfile:
      data = json.dumps(y_dict, indent=4, sort_keys=True)
      outfile.write(data)
