"""
COMP 4107 Assignment #2

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466
"""

from sklearn.datasets import fetch_mldata
import numpy as np

# The MNIST database contains a total of 70000 examples of handwritten digits of size 28x28 pixels, labeled from 0 to 9:

def load_data():
    mnist = fetch_mldata('MNIST original')
    mnist.data = mnist.data/255.0   #normalizing data

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    # format and create a training_dataset
    for training_image in mnist.data[:60000]:
        training_images.append(np.reshape(training_image, (784, 1)))

    for training_label in mnist.target[:60000]:
        training_labels.append(vector_form(training_label))

    training_dataset = [(image, label) for image, label in zip(training_images, training_labels)]

    # format and create a testing_dataset
    for testing_image in mnist.data[60000:]:
        testing_images.append(np.reshape(testing_image, (784, 1)))

    for testing_label in mnist.target[60000:]:
        testing_labels.append(vector_form(testing_label))

    testing_dataset = [(image, label) for image, label in zip(testing_images, testing_labels)]

    return (training_dataset, testing_dataset)

def vector_form(target):
    v = np.zeros((10, 1))
    v[int(target)] = 1.0
    return v
