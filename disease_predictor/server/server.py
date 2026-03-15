from flask import Flask, request, jsonify
from flask_cors import CORS
from softmax_logistic_regression_new.disease_predictor import predictor

server = Flask(__name__)
CORS(server)

@server.route("/softmax", methods=["POST"])
def softmax_predict():
    data = request.json
    res = predictor.predict(data)
    return jsonify(res.flatten().tolist())

server.run(host="0.0.0.0", port=5000)