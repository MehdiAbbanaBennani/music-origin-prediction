# # Paris
# coord_1 = [48.8566, 2.3522]
# c_paris = [to_radian(coord) for coord in coord_1]
# # London
# coord_london = [51.5074, - 0.1278]
# c_london = [to_radian(coord) for coord in coord_london]
# # Tokyo
# coord_tokyo = [35.6895, 139.6917]
# c_tokyo = [to_radian(coord) for coord in coord_tokyo]
# # NYC
# coord_nyc = [40.7128, - 74.0060]
# c_nyc = [to_radian(coord) for coord in coord_nyc]
#
# acos_dist(c_paris, c_london)
# acos_dist(c_paris, c_tokyo)
# acos_dist(c_london, c_tokyo)
# acos_dist(c_london, c_nyc)
# acos_dist(c_paris, c_nyc)

from acos_dist import *
from utils import to_radian

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  # Paris
  coord_1 = [48.8566, 2.3522]
  c_paris = [to_radian(coord) for coord in coord_1]
  # London
  coord_london = [51.5074, - 0.1278]
  c_london = [to_radian(coord) for coord in coord_london]



  x_paris = tf.constant(c_paris, tf.float32)
  x_london = tf.constant(c_london, tf.float32)
  y = acosdistance(x_london, x_paris)
  sess.run(init_op)
  print(y.eval())
  print(tf.gradients(y, [x_paris, x_london])[0].eval())