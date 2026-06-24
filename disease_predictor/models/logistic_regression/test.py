import pickle
import numpy as np
from .dataset import Dataset
# python -m disease_predictor.models.logistic_regression.test
def test_accuracy():
    train_pos_accuracy_avg = 0
    train_neg_accuracy_avg = 0
    train_pos_percent_avg = 0
    train_neg_percent_avg = 0

    test_pos_accuracy_avg = 0
    test_neg_accuracy_avg = 0
    test_pos_percent_avg = 0
    test_neg_percent_avg = 0

    train_dataset = Dataset(True)
    test_dataset = Dataset(False) 

    for disease in range(773):
        with open(f"./disease_predictor/models/logistic_regression/state/BCE/state_{disease}.pkl", "rb") as f:
            model = pickle.load(f)
        
        Y_hat_train = model(train_dataset.X)
        Y_hat_test = model(test_dataset.X)
        
        Y_pred_train = (Y_hat_train >= 0.5).astype(int)
        train_pos_percent = np.mean(Y_hat_train[train_dataset.Y[disease] == 1])
        train_neg_percent = np.mean(Y_hat_train[train_dataset.Y[disease] == 0])
        train_pos_accuracy = np.mean(Y_pred_train[train_dataset.Y[disease] == 1] == 1)
        train_neg_accuracy = np.mean(Y_pred_train[train_dataset.Y[disease] == 0] == 0)
        train_accuracy = (train_pos_accuracy + train_neg_accuracy) / 2
        
        Y_pred_test = (Y_hat_test >= 0.5).astype(int)
        test_pos_percent = np.mean(Y_hat_test[test_dataset.Y[disease] == 1])
        test_neg_percent = np.mean(Y_hat_test[test_dataset.Y[disease] == 0])
        test_pos_accuracy = np.mean(Y_pred_test[test_dataset.Y[disease] == 1] == 1)
        test_neg_accuracy = np.mean(Y_pred_test[test_dataset.Y[disease] == 0] == 0)
        test_accuracy = (test_pos_accuracy + test_neg_accuracy) / 2
        
        train_pos_accuracy_avg += train_pos_accuracy
        train_neg_accuracy_avg += train_neg_accuracy
        train_pos_percent_avg += train_pos_percent
        train_neg_percent_avg += train_neg_percent
        
        test_pos_accuracy_avg += test_pos_accuracy
        test_neg_accuracy_avg += test_neg_accuracy
        test_pos_percent_avg += test_pos_percent
        test_neg_percent_avg += test_neg_percent
        
    train_pos_accuracy_avg /= 773
    train_neg_accuracy_avg /= 773
    train_pos_percent_avg /= 773
    train_neg_percent_avg /= 773

    test_pos_accuracy_avg /= 773
    test_neg_accuracy_avg /= 773
    test_pos_percent_avg /= 773
    test_neg_percent_avg /= 773
        
    print(f"Train Positive Accuracy : {train_pos_accuracy}")
    print(f"Train Negative Accuracy : {train_neg_accuracy}")
    print(f"Train Positive Percent : {train_pos_percent}")
    print(f"Train Negative Percent : {train_neg_percent}")

    print(f"Test Positive Accuracy : {test_pos_accuracy}")
    print(f"Test Negative Accuracy : {test_neg_accuracy}")
    print(f"Test Positive Percent : {test_pos_percent}")
    print(f"Test Negative Percent : {test_neg_percent}")

test_accuracy()
    
    
    