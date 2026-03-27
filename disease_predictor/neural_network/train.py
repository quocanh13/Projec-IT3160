import numpy as np
from neural_network.get_data import training_data, class_weight, test_data
from neural_network.neural_network_softmax import NeuralNetwork

def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


nn = NeuralNetwork(
    [
        (300, 377, relu, d_relu),
        (200, 300, relu, d_relu),
        (772, 200, softmax, None)
    ]
)

nn.load_weight()
nn.train(training_data, class_weight, 10, 0.1)