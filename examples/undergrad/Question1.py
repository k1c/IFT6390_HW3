"""
COMP 4107 Assignment #2

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 1

Develop a feed forward neural network in python that classifies the images found in the MNIST dataset.

You are to train your neural network using backpropagation. You must show that you have:
    Performed K-fold cross correlation.
    Used weight decay for regularization.
    Investigated the performance of your neural network for different (a) numbers of hidden layers and (b) size of hidden layers."""

import MNIST_Loader
import FFNetwork

#loadMNISTData() returns a dataset which is a list of length 70,000 containing (image, label) tuples:
    #image is the input data in the form of a 784x1 vector where each element is a normalized greyscale value
    #label is the target data in the form of a 10x1 vector with a 1 in the index of the target value
training_dataset, testing_dataset = MNIST_Loader.load_data()

#number and size of layers
layers = [784,30,10]

#hyper-parameters
learning_rate = 0.1
epochs = 10
lmbda = 2.0
num_folds = 5

scores = FFNetwork.k_fold_cross_validation(training_dataset, num_folds, layers, learning_rate, epochs, lmbda)
print('\nMean Classification Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

print("\nTesting with MNIST testing dataset")

network = FFNetwork.Network(layers)
network.stochastic_gradient_descent(training_dataset, learning_rate, epochs, lmbda)
print('\nNetwork Accuracy: %.3f%%' % ((network.network_accuracy(testing_dataset)/len(testing_dataset)) * 100))
