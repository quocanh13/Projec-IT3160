import csv
import math
import json
from data.data_creation import data_info, symptoms_list
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

def get_weight(type: Literal["softmax", "logistic", "softmax_new"]) -> list[dict["weight" : list[list[float]], "bias" : float]]:
    with open(f"./data/{type}_weight.json") as file:
        data = json.load(file)
        return data

def check(i: int, j: int):
    s = 0
    c_i = 0
    c_j = 0
    for k in range(377):
        if(test_data[i][0][k] == 1):
            c_i += 1
        if(test_data[j][0][k] == 1):
            c_j += 1
        if(test_data[i][0][k] == test_data[j][0][k] == 1):
            s += 1
    print(c_i, c_j, s)

def get_symtom(i : int):
    print(data_info[i])
    for k in range(377):
        if(train_data[i][0][k] == 1):
            print(symptoms_list[k][0], symptoms_list[k][1])
    
# train_data, test_data = get_data_logistic()

# get_symtom(481)
    
# check(501, 1)
# print(train_data[3][0])
# print(train_data[210][0])