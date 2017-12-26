from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set up Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10

# define input training function using the high level wrappers
# This goes as the input to the model_fn
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':mnist.train.images},
    y=mnist.train.labels,
    batch_size=batch_size, num_epochs = None, shuffle=True
)

# This is used to create the model used in the model_fn
def neural_net(x_dict):
    # Uses dictionary to avoid multiple inputs
    x = x_dict['images']
    layer_1 = tf.layers.dense(x,n_hidden_1)
    layer_2 = tf.layers.dense(n_hidden_1,n_hidden_2)
    out_layer = tf.layers.dense(n_hidden_2,num_classes)
    return out_layer

def model_fn(features, labels, mode):
    # Create the model
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits,1)
    pred_probas = tf.nn.softmax(logits)

    # Return back when in training mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_classes)

    # Define Loss and Optimizations
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels,dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())

    # Get accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)

    # Not sure what this is, need to check it out!
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(mode=mode,predictions=pred_classes,loss=loss_op,train_op=train_op,eval_metric_ops={'accuracy': acc_op})

    return estim_specs


model = tf.estimator.Estimator(model_fn=model_fn)

model.train(input_fn,steps=num_steps)


# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
model.evaluate(input_fn)


# Predict single images
n_images = 4
# Get images from test set
test_images = mnist.test.images[:n_images]
# Prepare the input data
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
# Use the model to predict the images class
preds = list(model.predict(input_fn))

# Display
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction:", preds[i])