from Experiment import Experiment
from Map import Map
from NeuralNet import NeuralNet


def run_experiment():
  # hyperparameters_lists = {"middle_layer_sizes": [5, 10, 20, 30, 50],
  #                          "activations": ["relu", "tanh"],
  #                          "step_sizes": [0.001, 0.01, 0.1],
  #                          "L2_regs": [0.1, 1, 10, 100],
  #                          "w_vars": [0.01, 0.1, 1]}
  hyperparameters_lists = {"middle_size": [30, 50],
                           "activation": ["tanh"],
                           "step_size": [0.1],
                           "L2_reg": [0.1],
                           "w_var": [0.01],
                           'loss_type': ["None"],
                           'y_type': ["coord"]}
  experiment = Experiment(hyperparameters_lists)
  experiment.run()


def run_neural_net():
  hyperparameters = {"middle_size": 8,
                     "L2_reg": 10,
                     "activation": 'tanh',
                     'step_size': 0.01,
                     'y_type': "coord",
                     'w_var': 0.05,
                     'loss_type': "None"}
  neural_net = NeuralNet(hyperparameters)
  neural_net.run()
  neural_net.store_predictions()


def run_kl_neural_net():
  hyperparameters = {"middle_size": 20,
                     "L2_reg": 0.1,
                     "step_size": 0.01,
                     "activation": "None",
                     "y_type": "heatmap",
                     "w_var": 0.05,
                     "loss_type": "kl"}
  neural_net = NeuralNet(hyperparameters)
  neural_net.run()


def plot_map():
  map = Map()
  map.run("train")


if __name__ == '__main__':
  run_experiment()
  # run_neural_net()
  # run_kl_neural_net()
  # plot_map()
