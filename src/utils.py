from math import pi

import autograd.numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# TODO : rescale the data

def extract_data(data_dir):
  """ Splits X and y and converts y to radian"""
  df = pd.read_csv(data_dir)
  data = df.as_matrix()
  X = data[:, :-2]
  y = np.array(to_radian(data[:, -2:]))
  return {"X": X,
          "y": y}


def to_radian(deg):
  return deg / 180 * pi


def rescale(y):
  y_rad = to_radian(y)
  return y_rad / pi


to_radian = np.vectorize(to_radian)
rescale = np.vectorize(rescale)


def load_data(data_dir, test_size):
  data = extract_data(data_dir)
  X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"],
                                                      test_size=test_size,
                                                      random_state=42)
  return X_train, X_test, y_train, y_test
