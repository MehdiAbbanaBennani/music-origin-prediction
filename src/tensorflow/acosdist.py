from tensorflow import cos, sin, sqrt

import tensorflow as tf
from parameters import R
import math
import numpy as np


def acos_dist(coord_1, coord_2, R=R):
  lat_1 = coord_1[0]
  lat_2 = coord_2[0]
  long_1 = coord_1[1]
  long_2 = coord_2[1]

  f = math.sin((lat_2 - lat_1) / 2) ** 2 + math.cos(lat_1) * math.cos(lat_2) * math.sin(
    (long_1 - long_2) / 2) ** 2
  return np.float32(2 * R * math.asin(math.sqrt(f)))


vec_acos_dist = np.vectorize(acos_dist)


@tf.RegisterGradient("acos_distance")
def acos_dist_grad(op, grad):
  coord_1 = op.inputs[0]
  coord_2 = op.inputs[1]
  lat_1 = coord_1[0]
  lat_2 = coord_2[0]
  long_1 = coord_1[1]
  long_2 = coord_2[1]

  f = sin((lat_2 - lat_1) / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(
      (long_1 - long_2) / 2) ** 2
  common = R / sqrt(f * (1 - f ** 2))

  #   Delta partial derivative
  p_delta = sin((lat_2 - lat_1) / 2) * cos((lat_2 - lat_1) / 2) - sin(
      lat_1) * cos(lat_2) * sin(
      (long_1 - long_2) / 2) ** 2
  p_delta *= common

  # Lambda partial derivative
  p_lambda = cos(lat_1) * cos(lat_2) * sin((long_1 - long_2) / 2) * cos(
      (long_1 - long_2) / 2)
  p_lambda *= common

  return grad * p_delta, grad * p_lambda


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
  # Need to generate a unique name to avoid duplicates:
  rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1E+8)))

  tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc": rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def acosdistance(x, y, name=None):
  with tf.name_scope(name, "acos_distance", [x, y]) as name:
    f = py_func(vec_acos_dist,
                    [x, y],
                    [tf.float32],
                    name=name,
                    grad=acos_dist_grad)
    return f[0]


