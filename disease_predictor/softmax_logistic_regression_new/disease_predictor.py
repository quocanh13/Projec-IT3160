from softmax_logistic_regression_new.softmax_logsictic_regression import SoftmaxLogisticRegression
import numpy as np
from data.data_creation import mean, std
from data.get_data import get_weight

param = get_weight("softmax_new")
predictor = SoftmaxLogisticRegression(
    772, 
    377,
    np.array(mean),
    np.array(std),
    np.array(param["weight"]), 
    np.array(param["bias"])
)
del param