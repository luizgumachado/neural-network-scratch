import numpy as np

class NeuralNetwork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(weight.dot(a) + bias)
        return a

    def cost_derivative(self, output, y):
        return output - y

    def backprop(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        activation = x      # input activation vector
        activations = [x]   # list containing activation vectors of all layers
        z_vector = []       # list containg z values of all non-input layers (2, 3...)

        for bias, weight in zip(self.biases, self.weights):
            z = weight.dot(activation) + bias
            z_vector.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vector[-1]) # delta = output layer error
        gradient_b[-1] = delta                          # dC/dB = deltaL
        gradient_w[-1] = delta.dot(activations[-2].T)   # dC/dW = deltaL x a(L-1)

        for l in range(2, self.num_layers):
            z = z_vector[-l]
            sp = sigmoid_prime(z)
            delta = self.weights[-l + 1].T.dot(delta) * sp  # delta for layer l = (transpose of weights in l + 1 * delta) hadamard sigmoid prime of z
            gradient_b[-l] = delta
            gradient_w[-l] = delta.dot(activations[-l-1].T) # The gradient for the weights in the l layer = a(in) * delta(out)

        return(gradient_b, gradient_w)

    def update_batch(self, batch, alpha):
        # proper gradient descent for a single iteration through the batch
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            del_gradient_b, del_gradient_w = self.backprop(x, y)
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, del_gradient_b)]
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, del_gradient_w)]

            self.weights = [w - alpha/len(batch) * nw for w, nw in zip(self.weights, gradient_w)]
            self.biases = [b - alpha/len(batch) * nb for b, nb in zip(self.biases, gradient_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / 100

    def stochastic_gd(self, training_data, epochs, batch_size, alpha, test_data = None):
        if test_data: n_test = len(test_data)
        n = len(training_data)

        for i in range(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]

            for batch in batches:
                self.update_batch(batch, alpha)
            
            if test_data:
                print(f"Epoch {i + 1}: {self.evaluate(test_data)}%")
            else:
                print(f"Epoch {i + 1} complete")

def sigmoid(z):
            return 1.0/(1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))