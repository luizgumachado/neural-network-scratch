import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.NeuralNetwork([784, 30, 30, 30, 10])
net.stochastic_gd(list(training_data), 50, 10, 0.8, list(test_data))