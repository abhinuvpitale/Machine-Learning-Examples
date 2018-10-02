# Machine-Learning-Examples
Following the tutorials from [aymericdamien's Tensorflow Examples](https://github.com/aymericdamien/TensorFlow-Examples)

Reading material from [Neural Networks and deep Learning](http://neuralnetworksanddeeplearning.com/)

## 1: Your first models

Implement these with a basic knowledge of Data Analytics and Machine Learning

1. [Linear Regression](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb)
2. [Logistic Regression](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)
3. [kNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb)
4. [Random Forests (Needs a fix)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/random_forest.ipynb)
5. [CNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)

Also read the awesome tutorial on backpropogation from the same book ([Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html)) and why it works so well for neural nets

6. [CNN with Abstraction using Tensorflow (Needs a fix)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)

## 2: the Cross-entropy cost function and Regularization

Read what happens when the network has a slow learning rate due to the L2 cost function from [Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html)

1. Cross Entropy
2. SoftMax

Read about [Regularization](http://neuralnetworksanddeeplearning.com/chap3.html#regularization) and why it works. Simple is better, but not neccesarily.

3. Over Fitting
4. No Free Lunches

L1 Regularization - Makes the network smaller with lesser number of connections.
L2 Regularization - Makes sure the weights are not too big.
Dropout - Works similar to averaging multiple nets.

Data Augmentation - Makes the network more susceptible to changes

7. Convolution Neural Network (read chap 4)

8. Cat vs Dog - Tried an example using a vanilla CNN.

9. TensorBoard-Working - Example code for using tensorboard.

10. RNN(LSTM) - The timesteps are added to the LSTM cell.

Read this blog for more info on them
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Saving and Restoring a Model - Excellent blog post [here] (http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)

-----------------------------------------------------------------------------------------------------------------------------------------

Special Thanks to [Naresh](https://naresh1318.github.io/) for his awesome backpropogation for my errors!
