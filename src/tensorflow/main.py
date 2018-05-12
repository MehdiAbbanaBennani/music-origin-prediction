import sys
sys.path.extend(['/home/mehdi/PycharmProjects/music-origin-prediction/src'])

import tensorflow as tf
from sklearn.utils import shuffle
from utils import tf_acos_dist, extract_data
from parameters import *
from sklearn.model_selection import train_test_split

from acos_dist import acosdistance

# Import the parameters
if parameters["a_1_fun"] == "tanh" :
  a_1_fun = tf.tanh
if parameters["a_1_fun"] == "relu" :
  a_1_fun = tf.nn.relu
learning_rate = parameters["learning_rate"]
layers_dims = parameters["layer_dims"]
test_size = parameters["test_size"]

# Import and preprocess the data
data = extract_data(DATA_DIR)
X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"],
                                                    test_size = test_size,
                                                    random_state = 42)

# Build the tensorgraph
X = tf.placeholder("float", [None, layers_dims[0]])
y = tf.placeholder("float", [None, layers_dims[2]])

W_1 = tf.Variable(tf.random_normal([layers_dims[0], layers_dims[1]]))
W_2 = tf.Variable(tf.random_normal([layers_dims[1], layers_dims[2]]))
b_1 = tf.Variable(tf.random_normal([layers_dims[1]]))
b_2 = tf.Variable(tf.random_normal([layers_dims[2]]))

z_1 = tf.matmul(X, W_1) + b_1
a_1 = a_1_fun(z_1)
z_2 = tf.matmul(a_1, W_2) + b_2
a_2 = tf.tanh(z_2)

loss = tf.reduce_sum(acosdistance(a_2, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_examples = len(X_train)

  print("Training...")
  for i in range(EPOCHS):
    X_train_used, y_train_used = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
      end = offset + BATCH_SIZE
      batch_x, batch_y = X_train_used[offset:end], y_train_used[offset:end]
      sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
      sess.run(loss.eval(), feed_dict={X: batch_x, y: batch_y})
    # validation_loss = evaluate(X_valid, y_valid)
    print("EPOCH {} ...".format(i + 1))
    # print("Validation Accuracy = {:.3f}".format(validation_loss))
    # print()

  # saver.save(sess, './lenet')
  # print("Model saved")

