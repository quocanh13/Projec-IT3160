import numpy as np
import json
from datetime import datetime
from data.data_creation import  training_size, class_weight, test_size, mean, std
from data.get_data import get_weight, train_data, test_data
from softmax_logistic_regression_new.softmax_logsictic_regression import SoftmaxLogisticRegression

def create_X(data: list[list[list[float]]]) -> np.ndarray:
    X = [ [ value for value in sample ] for disease in data for sample in disease]
    return np.array(X)

def create_Y(is_train = True) -> np.ndarray:
    Y = np.zeros(training_size, dtype=int) if is_train else np.zeros(test_size, dtype=int)
    data = train_data if is_train else test_data
    start = 0
    for i in range(len(data)):
        length = len(data[i])
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
    param = get_weight("softmax_new")
    predictor = SoftmaxLogisticRegression(
        772, 
        377,
        np.array(mean),
        np.array(std),
        np.array(param["weight"]), 
        np.array(param["bias"])
    )
    predictor.train(X, Y, np.array(class_weight), epoch=30, alpha=100, print_res=True)
    if(write_weight):
        print("Writing")
        with open("./data/softmax_new_weight.json", "w") as file:
            json.dump({"weight" : predictor.get_weight(), "bias" : predictor.get_bias()}, file, indent=4)
        print("Wrote")

def test(size: list[tuple[int, int]] = []):
    param = get_weight("softmax_new")
    predictor = SoftmaxLogisticRegression(
        772,
        377,
        np.array(mean),
        np.array(std),
        np.array(param["weight"]), 
        np.array(param["bias"])
    )
    for (start, end) in size:
        for i in range(start, end + 1):
            res = predictor.predict(np.array(test_data[i][0]).reshape((1, 377)))
            m = np.argmax(res)
            print(i, res[0][i], m, res[0][m]) 
            
def test_accuracy():
    param = get_weight("softmax_new")
    predictor = SoftmaxLogisticRegression(
        772,
        377,
        np.array(mean),
        np.array(std),
        np.array(param["weight"]), 
        np.array(param["bias"])
    )
    X = create_X(test_data)
    Y = create_Y(False)
    res = predictor.predict(X)
    res = res.argmax(axis=1)
    return np.mean(res == Y)
start = datetime.now()

# train(write_weight=True)
# test([(0, 10), (100, 110), (500, 510)])
# test([(1, 1)])
# Y = create_Y()
# for i in range(Y.shape[0]):
#     print(i, Y[i])
print(test_accuracy())
end = datetime.now()
print(end - start)
