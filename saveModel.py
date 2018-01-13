import tensorflow as tf
import numpy

# Creating Placeholders, they will be restored later
input1 = tf.placeholder('float',name='input1')
input2 = tf.placeholder('float',name='input2')
weight = tf.Variable(2,dtype='float',name='weight')
output = tf.placeholder('float',name='output')

# Very important to give them names as they are used while restoring them
add_op = tf.add(input1,input2,name='add')
mul_op = tf.multiply(add_op,weight,name='multiply')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Get some random input
    in1 = 1.0
    in2 = 2.0

    add_output,mul_output = sess.run([add_op,mul_op],feed_dict={input1:in1,input2:in2})
    print(add_output,mul_output)


    # Create a saver
    saver = tf.train.Saver()
    # Saves the model in the path specifed in the second argument with global step as 10
    saver.save(sess,'./testSaver',global_step=10)