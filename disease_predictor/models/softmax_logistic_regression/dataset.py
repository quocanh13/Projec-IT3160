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

        Y_num = df["diseases"].to_numpy()
        self.X = df.drop(columns=["diseases"]).to_numpy()
        q, _ = self.X.shape
        self.Y = np.zeros((q, 773))
        self.Y[np.arange(q), Y_num] = 1
    
    
        
        