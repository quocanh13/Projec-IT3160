from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from neural_network.neural_network_softmax import predict as nn_predict
from softmax_logistic_regression_new.softmax_logsictic_regression import predict as sft_predict

server = Flask(__name__)
CORS(server)
# python -m server.server
@server.route("/predict", methods=["POST"])
def predict():
    #Hàm predict nhận vào một list là danh sách triệu chứng (0 / 1 : không / có triệu chứng) và trả về một list[float] là tỷ lệ các bệnh
    predict_map = {
        "softmax logistic" : sft_predict, 
        "neural network" : nn_predict,
        "naive bayes" : nn_predict,
        "svm" : nn_predict
    }

    data = request.json
    model = data["model"]
    symptom_list = data["symptomList"]
    return predict_map[model](symptom_list)

server.run(host="0.0.0.0", port=5100)