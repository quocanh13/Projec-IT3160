import numpy as np
import json
from logistic_regression import LogisticRegression
from data_creation import data_info, symptoms_list 
class Predictor:
    def __init__(self, weight: np.ndarray, bias: float, name: str, vname: str):
        self._model = LogisticRegression(weight, bias)
        self.name = name
        self.vname = vname

    def predict(self, X: list[float] | np.ndarray) -> float:
        return self._model.predict(X)[0]

class DiseasePredictor:
    def __init__(self):
        self._predictor: list[Predictor] = []
        self._symptoms: list[tuple] = symptoms_list
        with open("./weight.json") as file:
            weights = json.load(file)
            for i in range(len(data_info)):
                name, _, vname = data_info[i]
                self._predictor.append(Predictor(np.array(weights[i]["weight"]), weights[i]["bias"], name, vname))
        
    def predict(self, X: list[float] | np.ndarray, disease_num: int = 10) -> list[tuple[str, str, float]]:
        res = []
        for p in self._predictor:
            res.append((p.name, p.vname, p.predict(X)))
        res.sort(key=lambda x:x[2], reverse=True)
        return res[0:disease_num]
    
predictor = DiseasePredictor()
symptoms = [0]*377
symptoms[0] = 1.0
symptoms[27] = 1.0
symptoms[63] = 1.0
for (name, vname, pr) in predictor.predict(symptoms):
    print(f"{name} - {vname} - {pr}")