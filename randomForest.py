from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import resources

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
num_steps = 500 # Total steps to train the model
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10 # Number of trees to be built
max_nodes = 1000 # Max number of nodes per tree

# Input Graph
X = tf.placeholder(tf.float32,[None,num_features])
Y = tf.placeholder(tf.int32,[None,num_classes])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,num_features=num_features,num_trees=num_trees,max_nodes=max_nodes).fill()

# Build the Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Training Graphs and Loss Function
train_op = forest_graph.training_graph(X,Y)
loss_op = forest_graph.training_loss(X,Y)

# Accuracy
infer_op,_,_ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op,1),tf.cast(Y,tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Init
init_var = tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))

# Start a TensorFlow session
sess = tf.train.MonitoredSession()

# Run INit
sess.run(init_var)

# Train the model
for i in range(1,num_steps+1):
    # Get the Data
    if __name__ == '__main__':
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _,l = sess.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
        if i%50 == 0 or i == 1:
            acc = sess.run(accuracy_op,feed_dict={X:batch_x,Y:batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test the Model
test_x,test_y = mnist.test.images,mnist.test.labels
print("Test Accuracy :",sess.run(accuracy_op,feed_dict={X:test_x,Y:test_y}))
