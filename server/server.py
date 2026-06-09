from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from disease_predictor.models.logistic_regression import predict as logistic_regression_predict
from disease_predictor.models.knn import predict as knn_predict

server = Flask(__name__)
CORS(server)
# python -m server.server
@server.route("/predict", methods=["POST"])
def predict():
    #Hàm predict nhận vào một list là danh sách triệu chứng (0 / 1 : không / có triệu chứng) và trả về một list[float] là tỷ lệ các bệnh
    predict_map = {
        "logistic_regression" : logistic_regression_predict,
        "knn" : knn_predict
    }

    data = request.json
    model = data["model"]
    symptom_list = data["symptomList"]
    return predict_map[model](symptom_list)

server.run(host="0.0.0.0", port=5100)