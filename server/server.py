from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from disease_predictor.models.logistic_regression import predict as logistic_regression_predict
from disease_predictor.models.knn import predict as knn_predict
from disease_predictor.models.decision_tree import predict as decision_tree_predict
from disease_predictor.models.naive_bayes import predict as naive_bayes_predict
from disease_predictor.models.softmax_logistic_regression import predict as softmax_logistic_regression_predict
from disease_predictor.models.neural_network import predict as neural_network_predict

server = Flask(__name__)
CORS(server)
# python -m server.server
@server.route("/predict", methods=["POST"])
def predict():
    predict_map = {
        "logistic_regression" : logistic_regression_predict,
        "knn" : knn_predict,
        "decision_tree" : decision_tree_predict,
        "naive_bayes" : naive_bayes_predict,
        "neural_network" : neural_network_predict,
        "softmax_logistic_regression": softmax_logistic_regression_predict
    }

    data = request.json
    model = data["model"]
    symptom_list = data["symptomList"]
    return predict_map[model](symptom_list)

server.run(host="0.0.0.0", port=5100)