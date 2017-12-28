from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np

lib_path = os.path.abspath(os.path.join('./', 'catsVsDogs'))
sys.path.append(lib_path)
import fileIO

# Meow Woof
CAT = 1
DOG = 0

# set path
path = '../../ML Testers/catvsdog/train/train'

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10
dim = 3
# Network Parameters
image_size = 100
num_input = 30000
num_classes = 2
keep_prob = 0.75


# Input Variable and Graph
X = tf.placeholder(tf.float32,[None, image_size, image_size, dim])
Y = tf.placeholder(tf.uint8,[None,num_classes])
dropout = tf.placeholder(tf.float32)
# Generic Layers to be used
def conv2d(inputs, weights, biases, strides = 1):
    inputs = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, strides, strides, 1], padding='SAME')
    inputs = tf.nn.bias_add(value=inputs,bias=biases)
    return tf.nn.relu(inputs)


def maxpool2d(inputs, k=2):
    return tf.nn.max_pool(value = inputs, ksize = [1,k,k,1], strides = [1,k,k,1], padding='SAME')


# Create the Neural Network
def convnet(inputs,weights,biases,dropout):
    #inputs = tf.reshape(inputs,shape = [-1,100,100,3])

    # Layer1
    conv1 = conv2d(inputs,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)

    # Layer2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    # Fully Connectec Layer
    fc1 = tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1,dropout)

    # Output Layer
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

# Define Biases and Weights!
weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,3,32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([25*25*64,1024])),
    'out': tf.Variable(tf.random_normal([1024,num_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Creating the model
logits = convnet(X,weights,biases,keep_prob)
prediction = tf.nn.softmax(logits)

# Define Loss and Optimiser

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimiser = tf.train.AdamOptimizer(learning_rate)
train_op = optimiser.minimize(loss_op)

# Evaluate Model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initialiser
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1,num_steps+1):
        # Get input
        batch_x,batch_y = fileIO.get_next_batch(batch_size=batch_size,image_size=image_size)
        batch_y = tf.one_hot(np.reshape(batch_y, [-1]), 2)
        sess.run(train_op,feed_dict={X:batch_x,Y:sess.run(batch_y),dropout:keep_prob})
        if step%display_step == 0  or step == 1:
            loss,acc =  sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:sess.run(batch_y),dropout:1.0})
            print('Step '+str(step)+' Loss:'+str(loss)+' Accuracy: '+str(acc))
    print('Optimised!!')