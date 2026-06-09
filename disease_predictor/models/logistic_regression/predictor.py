import pickle
import numpy as np
from disease_predictor.models.logistic_regression import LogisticRegression

models : list[LogisticRegression] = []

for i in range(773):
    with open(f"./disease_predictor/models/logistic_regression/state/BCE/state_{i}.pkl", "rb") as f:
        model = pickle.load(f)
    models.append(model)

def predict(symptoms : list[int]) -> list[float]:
    symptoms = np.array(symptoms).reshape((1, 377))
    res = []
    for i in range(773):
        pred = models[i](symptoms)
        res.append(pred[0][0].item())
    
    return res
        

