import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# Graph Inputs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Weights
W = tf.Variable(rng.randn(),name = "Weight")
b = tf.Variable(rng.randn(),name = "Bais")

# Constructing the linear model
# Linear Regression
# Y = W*X + b
pred = tf.add(tf.multiply(X,W),b)

# Cost Function
# ((Y-pred).^2)/N
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

# Gradient descent optimisation
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Assign default values
init = tf.global_variables_initializer()

# Actual Training Starts here
with tf.Session() as sess:
    sess.run(init)

    # Use the training data to fit the model
    for epoch in range(training_epochs):
        # zip merges two lists of same length
        for (x,y) in zip(train_X,train_Y):
            # Pass elements zipped in (x,y) to the gradient optimiser
            sess.run(optimizer,feed_dict={X:x,Y:y})

        # Display the cost c and the optimised values every nth display step
        if (epoch+1)%display_step == 0:
            c = sess.run(cost,feed_dict={X:x,Y:y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W=", sess.run(W), "b=", sess.run(b))

    print("Done with the optimisation")
    training_cost = sess.run(cost, feed_dict={X:x,Y:y})
    print("Final Cost",training_cost,"W=",sess.run(W),"b=",sess.run(b))

    # Display Stuff
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()