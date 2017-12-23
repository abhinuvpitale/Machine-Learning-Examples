import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

# Train and Test on a limited set
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

# Variables
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Use L1 Distance to get the nearest distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
pred = tf.argmax(distance,0)

accuracy = 0

# Initialisation
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # for each training sample
    for i in range(len(Xte)):
        nn_index = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),"True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)