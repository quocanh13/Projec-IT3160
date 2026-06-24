import pickle
import logging
import numpy as np
import argparse
from datetime import datetime
from .model import *
from .dataset import Dataset
# python -m disease_predictor.models.softmax_logistic_regression.train
parser = argparse.ArgumentParser()
parser.add_argument("--lr", "-lr", type=float, default=0.01)
parser.add_argument("--epoch", "-ep", type=int, default=10)

args = parser.parse_args()

train_dataset = Dataset(True)
test_dataset = Dataset(False)    

# model = SoftmaxLogisticRegression(weight=np.ones((773, 377))*0.01, bias=np.zeros((1, 773)))
with open(f"./disease_predictor/models/softmax_logistic_regression/state.pkl", "rb") as f:
    model = pickle.load(f)


for i in range(args.epoch):
    start = datetime.now()

    Y_hat_train = model(train_dataset.X)
    Y_hat_test = model(test_dataset.X)
    loss = softmax_cross_entropy_loss(Y_hat_train, train_dataset.Y)
    dlw, dlb = sofmax_Cross_entropy_grad(train_dataset.X, Y_hat_train, train_dataset.Y)
    
    model.W -= dlw * args.lr
    model.B -= dlb * args.lr
    
    pred_train = np.argmax(Y_hat_train, axis=1)
    true_train = np.argmax(train_dataset.Y, axis=1)
    acc_train = np.mean(pred_train == true_train)
    percent_train = np.mean((Y_hat_train*train_dataset.Y).sum(axis=1))    

    pred_test = np.argmax(Y_hat_test, axis=1)
    true_test = np.argmax(test_dataset.Y, axis=1)
    acc_test = np.mean(pred_test == true_test)   
    percent_test = np.mean((Y_hat_test*test_dataset.Y).sum(axis=1))

    end = datetime.now()
    print(f"Ep: {i} - Time: {end - start} - Loss: {loss:.5f} - Accuracy Train: {acc_train:.5f} - Accuracy Test: {acc_test:.5f} - Percent Train: {percent_train:.5f} - Percent Test: {percent_test:.5f}")
    
with open(f"./disease_predictor/models/softmax_logistic_regression/state.pkl", "wb") as f:
    pickle.dump(model, f)

    
