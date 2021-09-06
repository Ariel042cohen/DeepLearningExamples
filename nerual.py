import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(1)


# Calculate the transfer function of a neuron
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calculate the derivative of a neuron output
def sigmoid_derivative(sig_val):
    return sig_val * (1 - sig_val)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    deriv = np.ones_like(x)
    deriv[x <= 0] = 0
    return deriv


def leaky_relu(x, eps=0.01):
    relu_output = np.maximum(x, eps * x)
    return relu_output


def leaky_relu_derivative(x, eps=0.01):
    deriv = np.ones_like(x)
    deriv[x <= 0] = eps
    return deriv


# one layer implementation

# initialize weights
def initialize_weights(input_dim):
    weights = 2 * np.random.random((input_dim, 1)) - 1
    # weights *= 0.01
    bias = 0
    # weights = np.zeros(3,1) - 1
    return weights, bias


def one_layer_neural_network(x, y, lr=1, epochs=10000, activation='sigmoid'):
    input_dim = x.shape[1]
    activation_func = activation_dict[activation]
    activation_deriv_func = activation_deriv_dict[activation]
    weights, bias = initialize_weights(input_dim)
    losses = []
    # full batch gradient descent
    for i in range(epochs):
        # forward pass
        res = x @ weights + bias
        l1 = activation_func(res)
        error = l1.T - y
        loss = 0.5 * np.sum(error ** 2)

        # backward pass (calculating derivatives)
        l1_delta = error.T * activation_deriv_func(l1)
        deriv_w = x.T @ l1_delta
        deriv_b = l1_delta.sum()

        # gradient descent
        weights -= lr * deriv_w
        bias -= lr * deriv_b

        losses.append(loss)
    return losses, l1


## Two layers implementation

# initialize weights
def initialize_weights_two_layer(input_dim, num_neurons):
    w0 = 2 * np.random.random((input_dim, num_neurons)) - 1
    w1 = 2 * np.random.random((num_neurons, 1)) - 1
    # alpha = np.random.rand(1)
    # beta = np.random.rand(1)
    b0 = np.zeros(num_neurons)
    b1 = 0
    return w0, w1, b0, b1


def two_layer_neural_network(x, y, num_neurons=3, lr=1, epochs=10000, activation='sigmoid'):
    input_dim = x.shape[1]
    w0, w1, b0, b1 = initialize_weights_two_layer(input_dim, num_neurons)
    activation_func = activation_dict[activation]
    activation_deriv_func = activation_deriv_dict[activation]

    losses = []
    for i in range(epochs):
        # forward pass
        l1 = activation_func(x @ w0 + x @ (w0*w0) + b0)
        l2 = activation_func(l1 @ w1 + b1)
        l2_error = l2.T - y
        loss = 0.5 * np.sum(l2_error ** 2)

        # backward pass (calculating derivatives)
        l2_delta = l2_error.T * activation_deriv_func(l2)
        deriv_w1 = np.dot(l1.T, l2_delta)
        deriv_b1 = l2_delta.sum()

        l1_error = w1 @ l2_delta.T
        l1_delta = l1_error * activation_deriv_func(l1).T
        #deriv_w0 = x.T @ (l1_delta.T)
        # Answer for the question
        deriv_w0 = (x.T + 2 * w0 @ x.T) @ (l1_delta.T)
        deriv_b0 = l1_delta.sum(axis=1)

        # gradient descent
        w1 -= lr * deriv_w1
        b1 -= lr * deriv_b1
        w0 -= lr * deriv_w0
        b0 -= lr * deriv_b0

        losses.append(loss)
    return losses, l2


def plot_losses(losses, num_layers_string):
    plt.plot(losses)
    plt.title(num_layers_string + ' layer neural network')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    activation_dict = {'sigmoid': sigmoid, 'relu': relu, 'leaky-relu': leaky_relu}
    activation_deriv_dict = {'sigmoid': sigmoid_derivative, 'relu': relu_derivative, \
                             'leaky-relu': leaky_relu_derivative}

    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([0, 0, 1, 1])
    losses, l1 = one_layer_neural_network(x, y, epochs=1000, activation='leaky-relu', lr=0.1)
    print(l1)
    plot_losses(losses, 'one')

    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([0, 1, 1, 0])
    losses, l2 = two_layer_neural_network(x, y, activation='relu', lr=0.1)
    print(l2)
    plot_losses(losses, 'two')