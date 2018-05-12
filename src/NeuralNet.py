import autograd.numpy as np
from utils import load_data
from autograd import grad
from neural_net import initialize_parameters, loss, error, reg_loss
from autograd.misc.optimizers import adam

from numpy import ceil
from parameters import DATA_DIR, TEST_SIZE, BATCH_SIZE, EPOCHS

class NeuralNet:
  def __init__(self, layer_sizes, L2_reg, activations, step_size):
    self.layer_sizes = layer_sizes
    self.L2_reg = L2_reg
    self.activations = activations
    self.step_size = step_size

  def run(self):
    L2_reg = self.L2_reg
    activations = self.activations
    step_size = self.step_size

    # Initial neural net parameters
    init_params = initialize_parameters(self.layer_sizes)

    print("Loading training data...")
    X_train, X_test, y_train, y_test = load_data(DATA_DIR, TEST_SIZE)
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

    optimized_params = adam(objective_grad,
                            init_params,
                            step_size=step_size,
                            num_iters=EPOCHS * num_batches,
                            callback=print_perf)
    return self.results(optimized_params, activations, L2_reg,
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