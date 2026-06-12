import json
import pandas as pd
import torch as tc
import torch.utils.data as torchdata

class Dataset(torchdata.Dataset):
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
        X = df.drop(columns=["diseases"]).to_numpy()
        self.X = tc.tensor(X, dtype=tc.float32)
        self.Y = tc.tensor(Y, dtype=tc.long)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        

        

