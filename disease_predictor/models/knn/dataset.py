import json
import pandas as pd
import numpy as np

df = pd.read_csv("./disease_predictor/dataset/dataset_1/data.csv")

Y = df.iloc[:, 0].to_numpy()                # cột đầu: tên bệnh
X = df.iloc[:, 1:].to_numpy(dtype=np.int8)  # các cột còn lại: triệu chứng (0/1)

with open("./disease_predictor/dataset/dataset_1/diseases.json") as f:
    diseases = json.load(f)

Y = np.array([diseases[disease] for disease in Y], dtype=np.int32)

# X: (247212, 377)  — 247k sample, 377 triệu chứng
# Y: (247212,)      — class index của từng bệnh
