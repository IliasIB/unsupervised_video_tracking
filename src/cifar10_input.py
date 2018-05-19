import numpy
import numpy as np
import pickle
import matplotlib.pyplot as plt


# https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py

def load_cifar10():
    # Load Training Data
    x_train = []
    y_train = []

    for i in range(1, 6):
        data_batch_path = '../cifar-10-batches-py/data_batch_' + str(i)
        data, labels = load_batch(data_batch_path)
        if i == 1:
            x_train = data
            y_train = labels
        else:
            x_train = np.vstack((x_train, data))
            y_train = np.vstack((y_train, labels))

    # Load Test Data
    test_batch_path = '../cifar-10-batches-py/test_batch'
    x_test, y_test = load_batch(test_batch_path)

    # Reshape Training data
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4)

    # Reshape Test Data
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4)

    y_train = y_train_reshape(y_train)
    y_test = y_test_reshape(y_test)

    return x_train, y_train, x_test, y_test


def y_train_reshape(y_train):
    flat_y_train = y_train.flatten()
    new_y_train = numpy.zeros(shape=(50000, 10))

    for count, element in enumerate(flat_y_train):
        new_y_train[count][element] = 1.0

    return new_y_train


def y_test_reshape(y_test):
    new_y_test = numpy.zeros(shape=(10000, 10))

    for count, element in enumerate(y_test):
        new_y_test[count][element] = 1.0

    return new_y_test


def load_batch(file_path):
    with open(file_path, 'rb') as file:
        dataframe = pickle.load(file)
    data = dataframe["data"]
    labels = dataframe["labels"]

    return data, labels
