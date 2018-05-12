import autograd.numpy as np
from utils import load_data
from autograd import grad
from neural_net import initialize_parameters, loss, error, \
  reg_loss, fully_connected
from autograd.misc.optimizers import adam
from Coordinates import Coordinates
import json

from numpy import ceil
from parameters import DATA_DIR, TEST_SIZE, BATCH_SIZE, EPOCHS, PREDICT_FILE


class NeuralNet:
  def __init__(self, layer_sizes, L2_reg, activations, step_size):
    self.layer_sizes = layer_sizes
    self.L2_reg = L2_reg
    self.activations = activations
    self.step_size = step_size
    self.Coordinates = None

    self.optimized_params = None

  def run(self):
    L2_reg = self.L2_reg
    activations = self.activations
    step_size = self.step_size

    # Initial neural net parameters
    init_params = initialize_parameters(self.layer_sizes)

    print("Loading training data...")
    X_train, X_test, y_train, y_test = load_data(DATA_DIR, TEST_SIZE)
    self.store(X_train, X_test, y_train, y_test)
    self.Coordinates = Coordinates(np.concatenate((y_train, y_test), axis=0))
    num_batches = int(ceil(X_train.shape[0] / BATCH_SIZE))

    def batch_indices(iter):
      idx = iter % num_batches
      return slice(idx * BATCH_SIZE, (idx + 1) * BATCH_SIZE)

    def objective(parameters, iter):
      idx = batch_indices(iter)
      return loss(parameters, X_train[idx], y_train[idx], L2_reg, activations)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    def print_perf(parameters, iter, gradient):
      if iter % num_batches == 0:
        train_acc = error(parameters, X_train, y_train, activations)
        test_acc = error(parameters, X_test, y_test, activations)
        train_reg = reg_loss(parameters, L2_reg)
        test_reg = reg_loss(parameters, L2_reg)
        print(
            "{:15}|{:20}|{:20}|{:20}|{:20}".format(iter // num_batches,
                                                   train_acc, test_acc,
                                                   train_reg, test_reg))

    self.optimized_params = adam(objective_grad,
                            init_params,
                            step_size=step_size,
                            num_iters=EPOCHS * num_batches,
                            callback=print_perf)
    return self.results(self.optimized_params, activations, L2_reg,
      X_train, X_test, y_train, y_test)

  def results(self, parameters, activations, L2_reg,
      X_train, X_test, y_train, y_test ):
    train_acc = error(parameters, X_train, y_train, activations)
    test_acc = error(parameters, X_test, y_test, activations)
    train_reg = reg_loss(parameters, L2_reg)
    test_reg = reg_loss(parameters, L2_reg)
    return {"train_acc" : train_acc,
            "test_acc" : test_acc,
            "train_reg" : train_reg,
            "test_reg" : test_reg
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

    y_dict = {"y_train_pred" : y_train_pred,
              "y_test_pred" : y_test_pred,
              "y_train" : y_train,
              "y_test" : y_test
              }

    with open(filename, 'w') as outfile:
      data = json.dumps(y_dict, indent=4, sort_keys=True)
      outfile.write(data)