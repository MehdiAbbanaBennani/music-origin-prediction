from math import pi

import autograd.numpy as np
import pandas as pd
from autograd.misc.flatten import flatten
from numpy.random import randint
from sklearn.model_selection import train_test_split

from parameters import COORD_DATA_DIR, HEATMAP_DATA_DIR, TEST_SIZE, EPSILON


def extract_heatmap_data(data_dir=HEATMAP_DATA_DIR):
  df = pd.read_csv(data_dir, sep='   ', lineterminator='\n', engine='python')
  data = df.as_matrix()
  y = np.array(data).astype(float)
  return y


def extract_data(y_type, data_dir=COORD_DATA_DIR):
  """ Splits X and y and converts y to radian"""
  df = pd.read_csv(data_dir)
  data = df.as_matrix()
  X = data[:, :-2]

  if y_type == "coord":
    y = np.array(to_radian(data[:, -2:]))
  if y_type == "heatmap":
    y = extract_heatmap_data()

  return {"X": X,
          "y": y}


def to_radian(deg):
  return deg / 180 * pi


def rescale(y):
  y_rad = to_radian(y)
  return y_rad / pi


to_radian = np.vectorize(to_radian)
rescale = np.vectorize(rescale)


def load_data(data_dir, test_size=TEST_SIZE):
  data = extract_data(data_dir)
  X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"],
                                                      test_size=test_size,
                                                      random_state=randint(0,
                                                                           42))
  return X_train, X_test, y_train, y_test


def relu(x):
  return np.maximum(0, x)


def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


# def kl_div(p, q):
#   p += EPSILON
#   q += EPSILON
#   return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_div(P,Q):
     epsilon = 0.0000001

     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P * np.log(P/Q))
     return divergence

def l2_norm(params):
  """Computes l2 norm of params by flattening them into a vector."""
  flattened, _ = flatten(params)
  return np.dot(flattened, flattened)
