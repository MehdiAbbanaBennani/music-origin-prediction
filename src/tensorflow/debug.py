import numpy as np
from math import cos, sin, sqrt, asin, pi
import tensorflow as tf
from tensorflow.python.framework import ops

def acos_dist(coord_1, coord_2, R=6371) :
  lat_1 = coord_1[0]
  lat_2 = coord_2[0]
  long_1 = coord_1[1]
  long_2 = coord_2[1]

  q_1 = sin((lat_2 - lat_1) / 2)**2
  q_2 = cos(lat_1) * cos(lat_2) * sin((long_1 - long_2)/2) ** 2
  return 2 * R * asin(sqrt(q_1 + q_2))


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
  # Need to generate a unique name to avoid duplicates:
  rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

  tf.RegisterGradient(rnd_name)(grad)
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc": rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def acos_dist_grad(op, grad):
  x = op.inputs[0]
  r = tf.mod(x, 1)
  n_gr = tf.to_float(tf.less_equal(r, 0.5))
  return grad * n_gr

def tf_acos_dist(x, name=None):
  with ops.op_scope([x], name, "acos_distance") as name:
    y = py_func(np_acos_dist_32,
                x,
                [tf.float32],
                name=name,
                grad=acos_dist_grad)  # <-- here's the call to the gradient
    return y[0]