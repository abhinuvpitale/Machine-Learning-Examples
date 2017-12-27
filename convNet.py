from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784
num_classes = 10
dropout = 0.75

# Create the input graph
X = tf.placeholder(tf.float32,[None,num_input])
Y = tf.placeholder(tf.float32,[None,num_classes])
keep_prob = tf.placeholder(tf.float32)

# Create some generic util functions
def conv2d(inputs, weights, biases, strides = 1):
    # Create a convolutional layer. Defined weights, stride and kept padding same to ensure same output shape as input.
    inputs = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, strides, strides, 1], padding='SAME')
    # Add the biases to the layer
    inputs = tf.nn.bias_add(value=inputs,bias=biases)
    # Add a ReLU layer to the convoluted output.
    return tf.nn.relu(inputs)

def maxpool2d(inputs, k=2):
    return tf.nn.max_pool(value = inputs, ksize = [1,k,k,1], strides = [1,k,k,1], padding='SAME')

# This is where you define the structure of your network. Which layer comes after which one!!
def conv_net(inputs, weights, biases, dropout):
    # reshape the input to match the MNIST image - > height*width*channels -> 28*28*1
    # tensor 4-d input should look like = [batch_size height width channel]

    x = tf.reshape(inputs, shape=[-1,28,28,1])

    # Layer 1
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool2d(conv2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply dropout
    fc1 = tf.nn.dropout(fc1,dropout)

    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out

# Define weights and biases. This step also determines the size of each layer!!

weights = {
    # conv1 = 5*5 conv, 1 input, 32 outputs
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    # conv2 = 5*5 conv, 32 inputs from the previous layer, 64 outputs
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    # fully connected layer, 7*7*64 inputs, 1024 outputs
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    # output layer, 1024 inputs, 10 outputs
    'out':tf.Variable(tf.random_normal([1024,num_classes]))
}

biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([num_classes]))
}

# Create the model
logits = conv_net(X,weights,biases,keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimiser functions
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate Model
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Initialiser
init = tf.global_variables_initializer()

# Actual Training of the model!!!!

with tf.Session() as sess:
    sess.run(init)

    for step in range(1,num_steps+1):
        # get random batch from the mnist database
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run the optimisation
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y,keep_prob:dropout})
        if step%display_step == 0 or step == 1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    print('Optimisation Finished')

    # Calculate accuracy for the test state:
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256],keep_prob: 1.0}))