import pickle
import numpy as np

with open(f"./disease_predictor/models/softmax_logistic_regression/state.pkl", "rb") as f:
    model = pickle.load(f)
    
def predict(symptoms: list[int]) -> list[float]:
    X = np.array(symptoms).reshape((1, -1))
    Y_hat = model(X)
    
    return Y_hat.flatten().tolist()
    