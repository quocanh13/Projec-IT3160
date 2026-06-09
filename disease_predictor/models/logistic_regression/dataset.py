import json
import pandas as pd
import numpy as np



class Dataset:
    DATA_PATH = {
        "train": "./disease_predictor/dataset/dataset_1/train_data.csv",
        "test": "./disease_predictor/dataset/dataset_1/test_data.csv"
    }
    
    INFO_PATH = {
        "disease": "./disease_predictor/dataset/dataset_1/diseases.json",
        "symptom": "./disease_predictor/dataset/dataset_1/symptoms.json"
    }
    
    def __init__(self, train=True):
        path = "train" if train else "test"
        df = pd.read_csv(self.DATA_PATH[path])

        with open(self.INFO_PATH["disease"]) as f:
            disease_info = json.load(f)

        df["diseases"] = df["diseases"].map(disease_info)

        Y = df["diseases"].to_numpy()
        self.X = df.drop(columns=["diseases"]).to_numpy()
        self.Y : list[np.ndarray] = []
        self.class_weight : list[np.ndarray] = []
        for disease in range(773):
            self.Y.append((Y == disease).astype(np.float32).reshape(-1, 1))
        
            pos = np.sum(self.Y[disease] == 1)
            neg = np.sum(self.Y[disease] == 0)

            w1 = (pos + neg) / pos
            w0 = (pos + neg) / neg
            
            self.class_weight.append(np.where(self.Y[disease] == 1, w1, w0))
        
        