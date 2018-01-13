import tensorflow as tf

with tf.Session() as sess:
    # Make sure you put the filepath correctly. It needs the .meta file to restore the graph
    saver = tf.train.import_meta_graph('./testSaver-10.meta')

    # This will restore the variables in the session
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    # Get the restored Graph, it is used to get the operations and other saved tensors
    currGraph = sess.graph

    # Get the input placeholders
    input1 = currGraph.get_tensor_by_name('input1:0')
    input2 = currGraph.get_tensor_by_name('input2:0')

    # Get the operations using their nameS!!
    mul_op = currGraph.get_tensor_by_name('multiply:0')

    in1 = 8
    in2 = 2

    # This is where the magic happens.
    # Without restoring the add operation, it still used it from the file and the operation used weight from the saved tensor !
    out = sess.run(mul_op,feed_dict={input1:in1,input2:in2})
    print(out)