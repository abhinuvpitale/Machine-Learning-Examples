import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

# Parameters

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])

# tf Model Weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct Model based on Logit function on xW+b
pred = tf.nn.softmax(tf.matmul(x,W)+b)

# Minimize the cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialise Variables
init = tf.global_variables_initializer()

# Training Starts here
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        # get the total number of possible batches
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # fit the model
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})

            avg_cost += c/total_batch
        if (epoch+1)%display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Done with Optimisation")

    # Test the Model
    # get all the matches, i.e. prediction == y (Note: pred is a matrix of N*10 dimensions where each of the 10 rows contain the probability of it being the output)
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy :",accuracy.eval({x:mnist.test.images[:3000],y:mnist.test.labels[:3000]}))