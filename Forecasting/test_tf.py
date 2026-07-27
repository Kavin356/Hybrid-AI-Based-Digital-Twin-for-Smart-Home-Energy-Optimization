import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print("Result:")
print(tf.matmul(a, b))
