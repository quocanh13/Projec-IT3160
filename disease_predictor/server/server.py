from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from neural_network.neural_network_softmax import NeuralNetwork

def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


nn_predictor = NeuralNetwork(
    [
        (300, 377, relu, d_relu),
        (200, 300, relu, d_relu),
        (772, 200, softmax, None)
    ]
)

nn_predictor.load_weight()

server = Flask(__name__)
CORS(server)

@server.route("/softmax", methods=["POST"])
def softmax_predict():
    data = request.json
    res = nn_predictor.predict(np.array(data).reshape((377, 1)))
    return jsonify(res.flatten().tolist())

server.run(host="0.0.0.0", port=5100)