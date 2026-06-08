import json
import pandas as pd
import numpy as np

df = pd.read_csv("./disease_predictor/dataset/dataset_1/test_data.csv")

Y = df.iloc[:, 0].to_numpy()               
X = df.iloc[:, 1:].to_numpy(dtype=np.int8)  

with open("./disease_predictor/dataset/dataset_1/diseases.json") as diseases_json:
    diseases = json.load(diseases_json)

Y = np.array([diseases[disease] for disease in Y], dtype=np.int32)