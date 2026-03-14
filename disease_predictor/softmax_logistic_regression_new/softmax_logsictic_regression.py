import numpy as np
from datetime import datetime
class SoftmaxLogisticRegression:
    @staticmethod
    def softmax_matrix(Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    @staticmethod
    def softmax(Z: np.ndarray):
        Z = Z.copy()
        max_z = np.max(Z)
        exp_Z = np.exp(Z - max_z)
        return exp_Z / np.sum(exp_Z)

    def __init__(self, m: int, n: int):
        self._W = np.zeros((m, n))
        self._B = np.zeros((m, 1))
        self._m = m
        self._n = n
        self._mean = np.zeros((n))
        self._std = np.zeros((n))
        self._total_sample = 0

    def train(
        self, 
        X: np.ndarray, 
        Y: np.ndarray,
        class_weight: np.ndarray, 
        epoch: int,
        alpha: float,
        epsilon: float = 1e-6,
        print_res = False
    ): 
        """

        Args:
            X (np.ndarray): 
            Y (np.ndarray): Y.shape = (q,)
            class_weight (np.ndarray): (m,)
            epoch (int): 
            alpha (float): 
            epsilon (float, optional): . Defaults to 10e-6.
        """
        m = self._m
        n = self._n
        W = self._W
        B = self._B
        q, x_col = X.shape
        y_len,  = Y.shape
        cw_row, = class_weight.shape

        if(x_col != n):
            raise Exception("Sai kích thước X")
        if(y_len != q):
            raise Exception("Sai kích thước Y")
        if(cw_row != m ):
            raise Exception("Sai kích thước class_weight")
        
        C = class_weight[Y].reshape((q, 1))
        
        total_sample = self._total_sample 
        mean = self._mean*total_sample + q*X.mean(axis=0)
        std = (self._std*total_sample + q*X.std(axis=0))
        total_sample += q
        mean /= total_sample
        std /= total_sample
        X = (X - mean) / (std + 1e-8) 
        
        self._mean = mean
        self._std = std
        self._total_sample = total_sample
        
        pre_loss = 1e9
        for i in range(epoch):
            start_time = datetime.now()
            
            P = self.softmax_matrix(X@W.T + B.T)
            loss = -(C.flatten() * np.log(P[np.arange(q), Y] + 1e-12)).mean()
            pred = np.argmax(P, axis=1)
            accuracy = np.mean(pred == Y)
            P[np.arange(q), Y] -= 1
            P *= C / q
            grad_W = P.T@X
            grad_B = (np.sum(P, axis=0, keepdims=True)).T
            
            self._W -= alpha*grad_W
            self._B -= alpha*grad_B
            end_time = datetime.now()
            
            if(print_res):
                print(f"epoch {i} --> Loss : {loss} - Accuracy : {accuracy} - Time : {end_time - start_time} --> {"good" if loss < pre_loss   else "bad" }")
            pre_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = (X - self._mean) / (self._std + 1e-8)
        return self.softmax_matrix(X @ self._W.T + self._B.T)

import numpy as np

# X = np.array([
#     [1.60, 60.0],
#     [1.70, 62.5],
#     [1.90, 80.7],
#     [1.63, 78.4],
#     [1.75, 92.0],
#     [1.81, 93.0],
#     [1.55, 55.0],
#     [1.68, 59.0],
#     [1.72, 65.0],
#     [1.85, 82.0],
#     [1.77, 75.0],
#     [1.80, 88.0],
#     [1.66, 72.0],
#     [1.73, 85.0],
#     [1.69, 90.0],
#     [1.82, 95.0]
# ])

# Y = np.array([
#     0, 0, 0, 1, 1, 1,
#     0, 0, 0, 0, 0, 1,
#     1, 1, 1, 1
# ])

# obese_predictor = SoftmaxLogisticRegression(2, 2)
# obese_predictor.train(X, Y, np.ones((2)), 5000, 30, print_res=True)
# print(obese_predictor.predict([1.9, 60]))
# print(obese_predictor.predict([1.3, 60]))
# a = np.array([5, 6, 7, 8])
# b = np.array([1, 1 , 2])
# c = np.ones((3, 1))



# print(a[b])