import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


def add_dims(X, n_dims):
    return np.concatenate((X, np.random.uniform(X.min(),
                                                X.max(),
                                                (X.shape[0], n_dims))), axis=1)

def softmax(model_output):
    e_x = np.exp(model_output - np.max(model_output, axis=0))
    return e_x / e_x.sum(axis=0)

def get_accuracy(pred, target):
    results = np.argmax(target, axis=0) == np.argmax(pred, axis=0)
    return len(results[results])/len(results)

def get_grads(target, pred, model_input, W, l1, l2, N):
    reg_term = l1 * np.sign(W) + 2 * l2 * W
    # We normalise the gradient with the batch size because otherwise we explode the gradient with big batches
    return (np.dot(model_input, pred.T) - np.dot(model_input, target.T)) / model_input.shape[1] + ((model_input.shape[1] * reg_term) / N)


def get_loss(target, pred, W, l1, l2):
    reg_term = l1 * np.sum(np.abs(W)) + l2 * np.sum(np.square(W))
    # We take the mean before taking the log to avoid exploding the loss with very small numbers
    return -np.log((target * pred + (1 - target) * (1 - pred)).mean()) + reg_term

def get_prediction(model_input, W):
    return softmax(np.dot(W.T, model_input))

def add_bias_term(X):
    return np.concatenate((X, np.ones((1, X.shape[1]))))

def get_mean(model_W, begin=0, end=-1):
    return np.mean(np.abs(model_W[begin:end]))

def get_var(model_W, begin=0, end=-1):
    return np.var(model_W[begin:end])

# Load data
digits = datasets.load_digits()
X = digits.data  # 8x8 image of a digit
y = digits.target  # int representing a target digit

# Add randomly distributed values at the end of the data
X = add_dims(X, n_dims=8)

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1

# Split train/test/valid
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test , test_size=0.5, random_state=42)

nb_epochs = 20
lrs = [0.01]
minibatch_sizes = [200]

lambda1 = 0.1
lambda2 = 1.e-8

# Grid search on learning rate and batch size
for lr in lrs:
    for minibatch_size in minibatch_sizes:

        legend = f'lr={lr}, bs={minibatch_size}'
        print(legend)
        legends.append(legend)

        # Initialize weights
        W = np.random.normal(0, 0.01, (len(np.unique(y)), (X.shape[1] + 1))).T  # weights of shape KxL

        accuracies = []
        losses = []
        valid_losses = []

        best_accuracy = 0
        # For each epoch
        for epoch in range(nb_epochs):

            # TRAINING
            loss = 0
            # For each batch
            for i in range(0, X_train.shape[0], minibatch_size):
                # Model input, target
                model_input = add_bias_term(X_train[i: i + minibatch_size].T)

                target = y_train[i: i + minibatch_size].T

                # Forward pass, get prediction
                pred = get_prediction(model_input, W)

                # Compute the loss
                loss += get_loss(target, pred, W, lambda1, lambda2)

                # Get the gradient of the loss
                grads = get_grads(target, pred, model_input, W, lambda1, lambda2, X_train.shape[0])

                # Get a step in the opposite direction
                delta = - lr * grads

                # update the weights
                W = W + delta

            losses.append(loss / ((X_train.shape[0] // minibatch_size) + 1 ))

            # VALIDATION
            loss = 0
            accuracy = 0
            # For each batch
            for i in range(0, X_validation.shape[0], minibatch_size):
                # Model input, target
                model_input = add_bias_term(X_validation[i: i + minibatch_size].T)

                target = y_validation[i: i + minibatch_size].T

                # Forward pass, get prediction
                pred = get_prediction(model_input, W)

                # Compute the loss and the accuracy
                loss += get_loss(target, pred, W, lambda1, lambda2)
                accuracy += get_accuracy(pred, target)

            valid_losses.append(loss / ((X_validation.shape[0] // minibatch_size) + 1))
            accuracies.append(accuracy / ((X_validation.shape[0] // minibatch_size) + 1))

            print(f'Loss {losses[-1]}, valid loss {valid_losses[-1]},  Accuracy {accuracies[-1]} at {epoch} epochs')
            # We keep a copy of the weights yielding to the best accuracy on the valid set
            if accuracy > best_accuracy:
                best_W = W.copy()
                best_accuracy = accuracy

        # TESTING
        pred = get_prediction(add_bias_term(X_test.T), best_W)
        accuracy_on_unseen_data = get_accuracy(pred, y_test.T)
        print(f'Test set Accuracy {accuracy_on_unseen_data}')

        mean_W_data = get_mean(best_W, begin=0, end=64)
        mean_W_noise = get_mean(best_W, begin=64, end=-1)
        var_W_data = get_var(best_W, begin=0, end=64)
        var_W_noise = get_var(best_W, begin=64, end=-1)
        print(f'DATA - mean(W) {mean_W_data}, var(W) {var_W_data}')
        print(f'NOISE - mean(W) {mean_W_noise}, var(W) {var_W_noise}')
