# Machine-Learning-Examples
Following the tutorials from [aymericdamien's Tensorflow Examples](https://github.com/aymericdamien/TensorFlow-Examples)

Reading material from [Neural Networks and deep Learning](http://neuralnetworksanddeeplearning.com/)

## 1: Your first models

Implement these with a basic knowledge of Data Analytics and Machine Learning

1. [Linear regression](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/linearRegression.py) 

[(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/linear_regression.ipynb)

2. [Logistic Regression](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/logisticRegression.py)

[(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)

3. [kNN](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/kNN.py)

[(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb)

4. [Random Forests (Needs a fix)](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/randomForest.py)

[(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/random_forest.ipynb)

5. [CNN](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/rawCNN.py)

[(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)

Also read the awesome tutorial on backpropogation from the same book ([Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html)) and why it works so well for neural nets

6. CNN with Abstraction using Tensorflow (Needs a fix) [(tutorial)](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)

## 2: the Cross-entropy cost function and Regularization

Read what happens when the network has a slow learning rate due to the L2 cost function from [Chapter 3](http://neuralnetworksanddeeplearning.com/chap3.html)

1. Cross Entropy
2. SoftMax

Read about [Regularization](http://neuralnetworksanddeeplearning.com/chap3.html#regularization) and why it works. Simple is better, but not neccesarily.

3. Over Fitting 

[(textbook)](http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization)

4. No Free Lunches

L1 Regularization - Makes the network smaller with lesser number of connections.
L2 Regularization - Makes sure the weights are not too big.
Dropout - Works similar to averaging multiple nets.

5. Data Augmentation: Makes the network more susceptible to changes

## 3: Deep delve into CNNs and RNNs

7. Convolution Neural Networks [(textbook chapter 4)](http://neuralnetworksanddeeplearning.com/chap4.html)

8. Cats vs Dogs - [Here's an example using a vanilla CNN](https://github.com/abhinuvpitale/Machine-Learning-Examples/tree/master/catsVsDogs)

9. [Example code for using tensorboard](https://github.com/abhinuvpitale/Machine-Learning-Examples/tree/master/Tensorboard%20working)

10. [RNN(LSTM)](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/rnn.py) 

The timesteps are added to the LSTM cell in the RNN example. For more info, read [this blog on Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

11. [Saving and Restoring a Model](https://github.com/abhinuvpitale/Machine-Learning-Examples/blob/master/saveModel.py)
[Here's an excellent blog post](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) on saving and restoring models 

-----------------------------------------------------------------------------------------------------------------------------------------

Special Thanks to [Naresh](https://naresh1318.github.io/) for his awesome backpropogation for my errors!
