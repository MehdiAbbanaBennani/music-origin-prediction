import itertools
import json

import autograd.numpy as np

from NeuralNet import NeuralNet
from parameters import LOG_FILE

class Experiment:
  def __init__(self, hyperparameters):
    self.logs = {"train_acc": [],
                 "test_acc": [],
                 "train_reg": [],
                 "test_reg": [],
                 "L2_reg": [],
                 "step_size": [],
                 "second_activation": [],
                 "middle_layer_size": []
                 }

    self.middle_layer_sizes = hyperparameters["middle_layer_sizes"]
    self.second_activations = hyperparameters["second_activations"]
    self.step_sizes = hyperparameters["step_sizes"]
    self.L2_regs = hyperparameters["L2_regs"]

  def log(self, log_dict, middle_layer_size, second_activation, step_size,
      L2_reg):
    self.logs["middle_layer_size"].append(middle_layer_size)
    self.logs["second_activation"].append(second_activation)
    self.logs["step_size"].append(step_size)
    self.logs["L2_reg"].append(L2_reg)
    for key, val in log_dict.items():
      self.logs[key].append(val)

  def log_to_file(self, log_file):
    with open(log_file, 'w') as outfile:
      data = json.dumps(self.logs, indent=4, sort_keys=True)
      json.dump(data, outfile)

  def run(self):
    hyperparameters_list = [self.middle_layer_sizes, self.second_activations,
                            self.step_sizes, self.L2_regs]
    for x in itertools.product(*hyperparameters_list):
      middle_layer_size, second_activation, step_size, L2_reg = x
      layer_sizes = [116, middle_layer_size, 2]
      activations = [np.tanh, second_activation]
      model = NeuralNet(layer_sizes, L2_reg, activations, step_size)
      log_dict = model.run()
      self.log(log_dict, middle_layer_size, second_activation, step_size, L2_reg)

    self.log_to_file(LOG_FILE)