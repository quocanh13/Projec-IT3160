import numpy as np
import math
import json
from datetime import datetime
from logistic_regression import LogisticRegression
from get_data import get_data_logistic, get_weight, train_data, test_data
from data_creation import symptoms_list, training_size, test_size, data_info

def train(start = 0, end = 0, epoch = 100, alpha = 0.1, file_name = ""):
    weights = get_weight()
    end += 1
    X = np.array([sample for disease in train_data for sample in disease])
    m, n = X.shape
    
    start_sample = 0
    for i in range(start):
        start_sample += len(train_data[i])

    
    for i in range(start, end):
        start_time = datetime.now()
        disease_predictor = LogisticRegression(np.array(weights[i]["weight"]), weights[i]["bias"])
        Y = np.zeros((m, 1))
        Y[start_sample : start_sample + len(train_data[i]), 0] = 1
        start_sample += len(train_data[i])
        weight_class = training_size / len(train_data[i])
        
        disease_predictor.train(X, Y, weight_class, epoch, alpha=alpha)
        weights[i]["weight"] = disease_predictor.get_weight()
        weights[i]["bias"] = disease_predictor.get_bias()
        with open(f"./weight{file_name}.json", "w") as file:
            print(f"Writing weights[{i}]")
            json.dump(weights, file, indent=4)
            print(f"Wrote weights[{i}]")
        end_time = datetime.now()
        print(end_time - start_time, end="\n\n")

def test(size: list[tuple[int, int]]):
    weights = get_weight()
    _, test = get_data_logistic()
    for (start, end) in size:
        end += 1
        for i in range(start, end):
            start_time = datetime.now()
            predictor = LogisticRegression(np.array(weights[i]["weight"]), weights[i]["bias"])
            pr0 = 0
            pr1 = 0
            for disease in range(0, i):
                for sample in test[disease]:
                    pr0 += math.fabs(predictor.predict(sample)[0])

            for sample in test[i]:
                pr1 += math.fabs(predictor.predict(sample)[0])
                
            for disease in range(i+1, len(test)):
                for sample in test[disease]:
                    pr0 += math.fabs(predictor.predict(sample)[0])
            end_time = datetime.now()
            print(end_time - start_time)
            pr0 = pr0/(test_size - len(test[i]))
            pr1 = pr1/len(test[i])
            pr = pr1 - pr0
            end_line = "\n"
            if(pr < 0.3): end_line = "<-------------------\n"
            print(f"{i} : {pr}", end=end_line)

def merge_weights():
    with open("./weight.json") as weights:
        weights_json = json.load(weights)
        size = [(0, 150), (151, 300), (301, 450), (451, 600), (601, len(data_info) - 1)]
        for i in range(5):
            with open(f"./weight_{i+1}.json") as file:
                data = json.load(file)
                start, end = size[i]
                weights_json[start:end+1] = data[start:end+1]
        with open("./weight.json", "w") as write_file:
            json.dump(weights_json, write_file, indent=4)

start = datetime.now()

train(0, 0, 10, 1000)
# test([(0, 10), (100, 110), (300, 310), (500, 510), (700, 710)])
# test([(0, 20), (200, 220), (400, 420), (700, 720)])

weights = get_weight()
predictor = LogisticRegression(np.array(weights[0]["weight"]), weights[0]["bias"])
print(predictor.predict(test_data[0][0]))

end = datetime.now()

print(end - start)