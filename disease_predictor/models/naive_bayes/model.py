import json
import numpy as np

class NaiveBayes:
    def __init__(
        self,
        symptom_size: int | None = None,
        disease_size: int | None = None,
        alpha: int | None = None
    ):
        if(symptom_size and symptom_size <= 0):
            raise ValueError("Symptom size must be greater than 0")
        if(disease_size and disease_size <= 0):
            raise ValueError("Disease size must be greater than 0")
        if(alpha and alpha <= 0):
            raise ValueError("Disease size must be greater than 0")
        
        self.symptom_size = symptom_size
        self.disease_size = disease_size
        self.alpha = alpha
        self.likelihood = np.zeros((disease_size, symptom_size))
        self.prior = np.zeros((disease_size))
        
    def fit(
        self, 
        X: np.ndarray,
        Y: np.ndarray
    ):
        n_samples = X.shape[0]
        for d in range(self.disease_size):
            X_d = X[Y == d]
            self.prior[d] = X_d.shape[0] / n_samples
            
            symptom_count = X_d.sum(axis=0)
            self.likelihood[d] = (symptom_count + self.alpha) / (X_d.shape[0] + self.symptom_size*self.alpha)
            
    def save_state(self, path: str):
        with open(path, "w") as state_file:
            json.dump({"likelihood" : self.likelihood, "prior" : self.prior}, state_file, indent=2)
    
    def load_state(self, path: str):
        with open(path) as state_file:
            state = json.load(state_file)
            self.likelihood = state["likelihood"]
            self.prior = state["prior"]

    def predict(
        self, 
        X: np.ndarray
    ):
        res = self.likelihood*X
        res = np.log(res)
        res = np.sum(res, axis=0) + self.prior
        return res