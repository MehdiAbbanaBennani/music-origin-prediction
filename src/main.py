import autograd.numpy as np
from NeuralNet import NeuralNet
# from Experiment import Experiment
from utils import relu
from Map import Map

if __name__ == '__main__':
  # neural_net = NeuralNet(layer_sizes=[116, 15, 2],
  #                        L2_reg=10,
  #                        activations=[np.tanh, relu],
  #                        step_size=0.01)
  # neural_net.run()
  # neural_net.store_predictions()

  # hyperparameters = {"middle_layer_sizes" : [10, 20, 30],
  #                    "second_activations" : [np.tanh, relu],
  #                    "step_sizes" : [0.001, 0.01, 0.1],
  #                    "L2_regs" : [0.1, 1, 10, 100]}
  map = Map()
  map.run("train")

  # experiment = Experiment(hyperparameters)
  # experiment.run()