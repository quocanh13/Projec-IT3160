import torch as tc
from .model import DiseasePredictor

model = DiseasePredictor()
model.load_state()

def predict(symptomList: list[int]) -> list[float]:
    X = tc.tensor(symptomList, dtype=tc.float).reshape((1, -1))
    proba = model.predict_proba(X)
    return proba.flatten().tolist()