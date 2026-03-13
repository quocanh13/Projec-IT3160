import csv
import math
import json
from data.data_creation import data_info
from typing import Literal

def get_data_logistic() -> tuple[list[list[float]], list[list[float]]]:
    """

    Returns:
        tuple[list[list[float]], list[list[float]]]: data_set[disease][sample]
    """
    with open("./data/training_data.csv") as training_data, \
         open("./data/test_data.csv") as test_data:
        training_csv = csv.reader(training_data)
        test_csv = csv.reader(test_data)
        training: list[list[float]] = []
        test: list[list[float]] = []
        
        next(training_csv); next(test_csv)
        for _, size, _ in data_info:
            training.append([])
            i = len(training) - 1
            for _ in range(math.floor(0.8*size)):
                row = next(training_csv)
                training[i].append([float(row[k]) for k in range(1, len(row))])
            
            test.append([])
            i = len(test) - 1
            for _ in range(math.floor(0.8*size), size):
                row = next(test_csv)
                test[i].append([float(row[k]) for k in range(1, len(row))])
        
        return training, test

def get_weight(type: Literal["softmax", "logistic"]) -> list[dict["weight" : list[list[float]], "bias" : float]]:
    with open(f"./data/{type}_weight.json") as file:
        data = json.load(file)
        return data

train_data, test_data = get_data_logistic()
# print(train_data[3][0])
# print(train_data[210][0])