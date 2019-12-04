import pickle
import numpy as np

import matplotlib.pyplot as plt

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims): # dims is of size 2 containing the input dimension and the number of classes
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            low = -1.0/np.sqrt(all_dims[layer_n - 1])
            high = 1.0/np.sqrt(all_dims[layer_n - 1])
            self.weights[f"W{layer_n}"] = np.random.uniform(low, high, (all_dims[layer_n - 1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n])) # no biases on input dimension

    def relu(self, x, grad=False):
        if grad:
            return (self.relu(x) > 0).astype(int)
        return np.maximum(0, x)

# source for numerically stable sigmoid: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    def sigmoid(self, x, grad=False):
        if grad:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        "Numerically stable sigmoid function."
        if x.all() >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def tanh(self, x, grad=False):
        if grad:
            return 1 - self.tanh(x) ** 2
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        if x.ndim > 1:
            e_x = np.exp(x - np.max(x, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
        else:
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x, axis=0)


    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        num_layers = self.n_hidden + 2
        Z = x
        for layer_n in range(1, num_layers):  # iterating through number of layers
            weights = self.weights[f"W{layer_n}"]
            biases = self.weights[f"b{layer_n}"]
            A = np.dot(Z, weights) + biases
            cache[f"A{layer_n}"] = A
            if layer_n == num_layers - 1:
                Z = self.softmax(A)
                cache[f"Z{layer_n}"] = Z
            else:
                Z = self.activation(A)
                cache[f"Z{layer_n}"] = Z
        return cache


    # def backward(self, cache, labels): #cache is from the forward function on a mini-batch
    #     output = cache[f"Z{self.n_hidden + 1}"]
    #     grad_a = - (labels - output)
    #     grads = {}
    #     print("range max",len(cache) - 5 )
    #     print("CACHE", cache.keys())
    #     for i in range(len(cache) - 5):
    #         index = len(cache) - i - 3
    #         print("index from back", index)
    #         grad_W = np.dot(grad_a.T, cache[f"Z{index - 3}"]).T
    #         grad_b = np.sum(grad_a, axis=0)[None, :]
    #         grads[f"dA{index-2}"] = grad_a
    #         grads[f"dW{index-2}"] = grad_W / float(labels.shape[0])
    #         grads[f"db{index-2}"] = grad_b / float(labels.shape[0])
    #         if index > 3:
    #             print(self.weights.keys())
    #             print("Weight",f"W{index-2}")
    #             grad_h = np.dot(grad_a, self.weights[f"W{index-2}"].T)
    #             grads[f"dZ{index-3}"] = grad_h
    #             grad_a = np.multiply(grad_h, self.activation(cache[f"A{index - 3}"], grad=True))
    #     return grads

    def backward(self, cache, labels):  # cache is from the forward function on a mini-batch
        output = cache[f"Z{self.n_hidden + 1}"]
        grad_a = - (labels - output)
        grads = {}
        num_layers = self.n_hidden + 2
        for i in range(num_layers - 1, 0, -1):
            grad_W = np.dot(grad_a.T, cache[f"Z{i - 1}"]).T
            grad_b = np.sum(grad_a, axis=0)[None, :]
            grads[f"dA{i}"] = grad_a
            grads[f"dW{i}"] = grad_W / float(labels.shape[0])
            grads[f"db{i}"] = grad_b / float(labels.shape[0])
            if i > 1:
                grad_h = np.dot(grad_a, self.weights[f"W{i}"].T)
                grads[f"dZ{i}"] = grad_h
                grad_a = np.multiply(grad_h, self.activation(cache[f"A{i - 1}"], grad=True))
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - (self.lr *  grads[f"dW{layer}"])
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - (self.lr * grads[f"db{layer}"])

    def one_hot(self, y):
        b = np.zeros((y.size, self.n_classes))
        b[np.arange(y.size), y] = 1
        return b

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        N = prediction.shape[0]
        ce = -np.sum(labels * np.log(prediction + 1e-9)) / N
        return ce

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid

            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy


def plot_curves(train, valid, epochs, metric_name, model_kwargs):
    t = np.arange(len(train))
    plt.ylabel(f'Average {metric_name}')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.xlim(0, epochs)
    plt.plot(t, train)
    plt.plot(t, valid)
    plt.title(f"Train and Valid {metric_name} on {epochs} epochs\n with {model_kwargs}")
    plt.legend(["train", "valid"], loc='upper right')
    plt.savefig(f"{metric_name}-{epochs}-{model_kwargs}.png", bbox_inches='tight')


def main(seed, hidden_dims):
    n_epochs = 50
    kwargs = {
        "seed": seed,
        "hidden_dims": hidden_dims,
        "lr": 0.003,
        "batch_size": 100,
    }
    model = NN(**kwargs)
    train_logs = model.train_loop(n_epochs=n_epochs)
    test_loss, test_accuracy = model.evaluate()

    print(kwargs)
    print(test_loss, test_accuracy)
    for metric_name in ["loss", "accuracy"]:
        plot_curves(train_logs[f'train_{metric_name}'],
                    train_logs[f'validation_{metric_name}'],
                    n_epochs,
                    metric_name,
                    kwargs)


if __name__ == '__main__':
    for seed in [0, 1, 2]:
        for hidden_dims in [(512, 256), (512, 120, 120, 120, 120, 120, 120, 120)]:
            main(hidden_dims=hidden_dims, seed=seed)
