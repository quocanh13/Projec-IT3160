import pickle
import logging
import numpy as np
import argparse
from datetime import datetime
from .model import *
from .dataset import Dataset
# python -m disease_predictor.models.softmax_logistic_regression.test

test_dataset = Dataset(False)    

# model = SoftmaxLogisticRegression(weight=np.ones((773, 377))*0.01, bias=np.zeros((1, 773)))
with open(f"./disease_predictor/models/softmax_logistic_regression/state.pkl", "rb") as f:
    model = pickle.load(f)

Y_hat_test = model(test_dataset.X)
 
pred_test = np.argmax(Y_hat_test, axis=1)
true_test = np.argmax(test_dataset.Y, axis=1)
acc_test = np.mean(pred_test == true_test)   
percent_test = np.mean((Y_hat_test*test_dataset.Y).sum(axis=1))
    
print(acc_test, percent_test)
    
