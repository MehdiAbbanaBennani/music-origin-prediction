import autograd.numpy as np
from utils import load_data
from autograd import grad
from neural_net import initialize_parameters, loss, error, reg_loss
from autograd.misc.optimizers import adam

from numpy import ceil
from parameters import DATA_DIR, TEST_SIZE, BATCH_SIZE, EPOCHS

if __name__ == '__main__':
  # Model parameters
  layer_sizes = [116, 20, 2]
  L2_reg = 1
  activations = [np.tanh, np.tanh]

  # Training parameters
  step_size = 0.001

  # Initial neural net parameters
  init_params = initialize_parameters(layer_sizes)

  print("Loading training data...")
  X_train, X_test, y_train, y_test = load_data(DATA_DIR, TEST_SIZE)
  # train_generator = batch_generator(X_train, y_train, BATCH_SIZE)
  # test_generator = batch_generator(X_test, y_test, BATCH_SIZE)
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