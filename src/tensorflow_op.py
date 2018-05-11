import tensorflow as tf
import numpy as np
from numpy import float32
from tensorflow.python.framework import ops


def f(x, y) :
  return tf.pow(x, 2) - 2 * tf.pow(y, 2)

@tf.RegisterGradient("distance")
def grad_f(op, grad):
  x = op.inputs[0]
  y = op.inputs[1]
  return grad * (2 * x), grad * (-4 * y)

with tf.Session() as sess :
  x = tf.constant([0., 1.])
  y = tf.constant([0., 5.])
  z = f(x, y)
  tf.initialize_all_variables().run()
  print(x.eval(), y.eval(), z.eval())
  print([tf.gradients(z, [x, y])[i].eval() for i in range(2)])
