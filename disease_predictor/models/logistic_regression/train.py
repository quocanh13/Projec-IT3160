import pickle
import logging
import numpy as np
from datetime import datetime
from .model import LogisticRegression, mse_grad, mse_loss, binary_cross_entropy_grad, binary_cross_entropy_loss
from .dataset import Dataset
# python -m disease_predictor.models.logistic_regression.train

logging.basicConfig(
    filename="./disease_predictor/models/logistic_regression/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_one(train_dataset: Dataset, test_dataset: Dataset, disease: int, epoch: int, lr: float):
    # model = LogisticRegression(weight=np.ones((377, 1))*0.01)
    with open(f"./disease_predictor/models/logistic_regression/state/BCE/state_{disease}.pkl", "rb") as f:
        model = pickle.load(f)

    for i in range(epoch):
        start = datetime.now()
        Y_hat_train = model(train_dataset.X)
        Y_hat_test = model(test_dataset.X)
        # dhw, dhb = mse_grad(train_dataset.X, Y_hat_train, train_dataset.Y[disease], train_dataset.class_weight[disease])
        dhw, dhb = binary_cross_entropy_grad(train_dataset.X, Y_hat_train, train_dataset.Y[disease], train_dataset.class_weight[disease])
        
        model.weight -= dhw*lr
        model.bias -= dhb*lr
        
        # loss = mse_loss(Y_hat_train, train_dataset.Y[disease])
        train_loss = binary_cross_entropy_loss(Y_hat_train, train_dataset.Y[disease], train_dataset.class_weight[disease])
        test_loss = binary_cross_entropy_loss(Y_hat_test, test_dataset.Y[disease], test_dataset.class_weight[disease])
        
        Y_pred_train = (Y_hat_train >= 0.5).astype(int)
        train_acc_class_1 = np.mean(Y_pred_train[train_dataset.Y[disease] == 1] == 1)
        train_acc_class_0 = np.mean(Y_pred_train[train_dataset.Y[disease] == 0] == 0)
        train_accuracy = (train_acc_class_1 + train_acc_class_0) / 2
        
        Y_pred_test = (Y_hat_test >= 0.5).astype(int)
        test_acc_class_1 = np.mean(Y_pred_test[test_dataset.Y[disease] == 1] == 1)
        test_acc_class_0 = np.mean(Y_pred_test[test_dataset.Y[disease] == 0] == 0)
        test_accuracy = (test_acc_class_1 + test_acc_class_0) / 2
        
        end = datetime.now()
        if(i % 10 == 9):
            logging.info(f"D : {disease} - Ep : {i} - TrainLoss : {train_loss:.5} - TestLoss : {test_loss:.5} - TrainPos : {train_acc_class_0} - TrainNeg : {train_acc_class_1} - TestPos : {test_acc_class_0} - TesNeg : {test_acc_class_1} - T : {end - start}")
        
    with open(f"./disease_predictor/models/logistic_regression/state/BCE/state_{disease}.pkl", "wb") as f:
        pickle.dump(model, f)

train_dataset = Dataset(True)
test_dataset = Dataset(False)    
for disease in range(773):
    start = datetime.now()
    train_one(train_dataset, test_dataset, disease, 50, 0.5)
    end = datetime.now()
    logging.info(f"D : {disease} -- Time : {end - start} \n")
    
