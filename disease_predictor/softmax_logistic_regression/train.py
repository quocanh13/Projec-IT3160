import numpy as np
import json
from datetime import datetime
from data.data_creation import test_size, training_size, symptoms_list, data_info, class_weight
from data.get_data import get_data_logistic, get_weight, train_data, test_data
from softmax_logistic_regression.softmax_logistic_regression import SoftmaxLogisticRegression

def create_X(data: list[list[list[float]]]) -> np.ndarray:
    X = [ [ value for value in sample ] for disease in data for sample in disease]
    return np.array(X)

def create_Y() -> np.ndarray:
    Y = np.zeros(training_size, dtype=int)
    start = 0
    for i in range(len(train_data)):
        length = len(train_data[i])
        Y[start : (start + length)] = i
        start += length
    return Y

def create_one_X(i: int):
    X = [ [value for value in sample] for sample in train_data[i]]
    return np.array(X)
def create_one_Y(i: int):
    length = len(train_data[i])
    return np.array([i]*length)
    

def train(write_weight = False):
    X = create_X(train_data)
    Y = create_Y()
    param = get_weight("softmax")
    predictor = SoftmaxLogisticRegression(
        np.array(param["weight"]), 
        np.array(param["bias"])
    )
    predictor.train(X, Y, class_weight=np.array(class_weight).reshape((772, 1)), epoch=10, alpha=50)
    if(write_weight):
        print("Writing")
        with open("./data/softmax_weight.json", "w") as file:
            json.dump({"weight" : predictor.get_weight(), "bias" : predictor.get_bias()}, file, indent=4)
        print("Wrote")

def test(size: list[tuple[int, int]] = []):
    param = get_weight("softmax")
    predictor = SoftmaxLogisticRegression(
        np.array(param["weight"]), 
        np.array(param["bias"])
    )
    for (start, end) in size:
        for i in range(start, end + 1):
            res = predictor.predict(np.array(test_data[i][0]).reshape((1, 377)))
            m = np.argmax(res)
            print(i, res[0][i], m, res[0][m])
    
start = datetime.now()

train(write_weight=True)
test([(0, 10), (100, 110), (500, 510)])
# Y = create_Y()
# for i in range(Y.shape[0]):
#     print(i, Y[i])

end = datetime.now()
print(end - start)
