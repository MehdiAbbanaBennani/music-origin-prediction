import itertools
import json
import datetime
from tqdm import tqdm

from NeuralNet import NeuralNet
from parameters import LOG_DIR


class Experiment:
  def __init__(self, hyperparameters_lists):
    self.logs = self.initialize_logs(hyperparameters_lists)
    self.hyperparameters_lists = hyperparameters_lists

    now = datetime.datetime.now().strftime("%b:%d:%Y:%H:%M:%S")
    self.log_file = LOG_DIR + "run_logs:" + now + ".json"

  @staticmethod
  def initialize_logs(hyperparameters_lists):
    results_keys = ["train_acc", "test_acc", "reg"]
    all_keys = list(hyperparameters_lists.keys()) + results_keys
    logs = {}
    for key in all_keys:
      logs[key] = []
    return logs

  def log(self, log_dict, hyperparameters):
    for key, val in log_dict.items():
      self.logs[key].append(val)
    for key, val in hyperparameters.items():
      self.logs[key].append(val)

  def log_to_file(self, log_file):
    with open(log_file, 'w') as outfile:
      data = json.dumps(self.logs, indent=4, sort_keys=True)
      outfile.write(data)

  def run(self):
    for hyperparameters in tqdm(self.dict_product(self.hyperparameters_lists)):
      model = NeuralNet(hyperparameters)
      log_dict = model.run()
      self.log(log_dict, hyperparameters)

    self.log_to_file(self.log_file)

  @staticmethod
  def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
