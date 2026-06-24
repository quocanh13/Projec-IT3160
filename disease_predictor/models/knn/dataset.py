import json
import pandas as pd
import numpy as np

df_train = pd.read_csv("./disease_predictor/dataset/dataset_1/train_data.csv")

Y_train = df_train.iloc[:, 0].to_numpy()             
X_train = df_train.iloc[:, 1:].to_numpy(dtype=np.int8) 

with open("./disease_predictor/dataset/dataset_1/diseases.json") as f:
    diseases = json.load(f)

Y_train = np.array([diseases[disease] for disease in Y_train], dtype=np.int32)



df_test = pd.read_csv("./disease_predictor/dataset/dataset_1/test_data.csv")

Y_test = df_test.iloc[:, 0].to_numpy()             
X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.int8) 

with open("./disease_predictor/dataset/dataset_1/diseases.json") as f:
    diseases = json.load(f)

Y_test = np.array([diseases[disease] for disease in Y_test], dtype=np.int32)