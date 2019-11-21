"""
COMP 4107 Assignment #2

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466
"""

import numpy as np
from math import exp
from random import shuffle
from random import randrange

class Network:

    def __init__(self, layers):

        self.layers = layers
        self.num_layers = len(layers)

        self.weights = self.init_weights(layers)
        self.biases = self.init_biases(layers)

    def init_weights(self, layers):
        network_weights = []

        for dim_row, dim_col in zip(layers[:-1], layers[1:]):
            weight_matrix = []
            for i in range(dim_col):
                row = []
                for j in range(dim_row):
                    row.append(np.random.randn())
                weight_matrix.append(row)
            network_weights.append(np.asarray(weight_matrix))

        return network_weights

    def init_biases(self, layers):
        network_biases = []
        for dim_row in layers[1:]:
            bias_vector = []
            for j in range(dim_row):
                row = []
                row.append(np.random.randn())
                bias_vector.append(row)
            network_biases.append(np.asarray(bias_vector))

        return network_biases

    def stochastic_gradient_descent(self, training_set, learning_rate, epochs, lmbda):

        for i in range(epochs):
            print("\nEpoch {} in progress...".format(i+1))

            # Shuffle the (image, label) tuples to prevent overfitting
            shuffle(training_set)

            for tuple_ in training_set:
                self.update_weights(tuple_, len(training_set), learning_rate, lmbda)

            print("Training Accuracy in Epoch", i+1, ": ", end="")
            print('%.3f%%' % ((self.network_accuracy(training_set)/len(training_set))*100))

    def update_weights(self, tuple_, training_set_size, learning_rate, lmbda):
        # call backpropagation on each tuple_
        image = tuple_[0]
        label = tuple_[1]
        weight_gradients, bias_gradients = self.backpropagation(image, label)

        # update weights and biases
        for k in range(len(self.weights)):
            self.weights[k] = (1 - ((learning_rate * lmbda) / training_set_size)) * self.weights[k] - (learning_rate * weight_gradients[k])

        for k in range(len(self.biases)):
            self.biases[k] -= learning_rate * bias_gradients[k]

    def backpropagation(self, image, label):

        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        # Feed Forward
        activations = []
        transfers = [image]

        for k in range(self.num_layers - 1):
            activations.append(np.dot(self.weights[k], transfers[k]) + self.biases[k])
            transfers.append(self.sigmoid(activations[k]))

        last_neuron_output = transfers[-1]

        # Back Propagation
        # Calculate the correction on last neuron
        last_neuron_error = self.cross_entropy_cost_prime(last_neuron_output, label)

        bias_gradients[-1] = last_neuron_error
        weight_gradients[-1] = np.dot(last_neuron_error, transfers[-2].transpose())

        # correct the other neurons in the hidden layers based on previous calculations
        hidden_neuron_error = last_neuron_error
        for k in range(2, self.num_layers):
            sp = self.sigmoid_prime(activations[-k])
            hidden_neuron_error = np.dot(self.weights[-k+1].transpose(), hidden_neuron_error) * sp
            bias_gradients[-k] = hidden_neuron_error
            weight_gradients[-k] = np.dot(hidden_neuron_error, transfers[-k-1].transpose())

        return (weight_gradients, bias_gradients)


    def feed_foward(self, image):
        activations = []
        transfers = [image]

        for k in range(self.num_layers - 1):
            activations.append(np.dot(self.weights[k],transfers[k]) + self.biases[k])
            transfers.append(self.sigmoid(activations[k]))

        return transfers[-1]

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def cross_entropy_cost_prime(self, output, label):
        return (output - label)

    def network_accuracy(self, testing_set):
        results = [(np.argmax(self.feed_foward(image)), np.argmax(label)) for (image, label) in testing_set]
        result_accuracy = sum(int(x == y) for (x, y) in results)

        return result_accuracy



def k_fold_cross_validation(dataset, num_folds, layers, learning_rate, epochs, lmbda):
    #Split dataset into k mutually exclusive subsets (folds)
    folds = split_dataset(dataset.copy(), num_folds)
    fold_accuracy = []
    for i in range(num_folds):
        print("\nTraining with folds ", end="")
        training_folds = []

        # Combine all folds into a single training set
        for j in range(num_folds):
            if (i != j):
                print(j+1, " ", end="")
                training_folds.append(folds[j])
        print("")
        print("Testing with fold", i+1)
        training_folds = sum(training_folds,[])
        testing_folds = folds[i]

        # initialize and train our network
        network = Network(layers)
        network.stochastic_gradient_descent(training_folds, learning_rate, epochs, lmbda)

        # test our network
        fold_accuracy.append((network.network_accuracy(testing_folds)/len(testing_folds))*100)
        print("\nClassification Accuracy in Fold ", i+1, ": ", end="")
        print("%.3f%%" % fold_accuracy[i])
        print("________________________________________")

    return fold_accuracy

def split_dataset(dataset, num_folds):
    folds = []
    fold_size = int(len(dataset)/num_folds)
    for i in range(num_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset))
            fold.append(dataset.pop(index))
        folds.append(fold)

    return folds
