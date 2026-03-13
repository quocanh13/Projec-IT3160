import numpy as np
import math
from datetime import datetime
class SoftmaxLogisticRegression:
    @staticmethod
    def softmax_matrix(Z: np.ndarray) -> np.ndarray:
        Z = Z.copy()
        q, _ = Z.shape
        for i in range(q):
            Z[i] = SoftmaxLogisticRegression.softmax(Z[i])
        return Z
    
    @staticmethod
    def softmax(Z: np.ndarray):
        Z = Z.copy()
        max_z = np.max(Z)
        exp_Z = np.exp(Z - max_z)
        return exp_Z / np.sum(exp_Z)
    
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self._W = weight.copy()
        self._B = bias.copy()
        
    def train(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        class_weight: np.ndarray, 
        epoch = 100, 
        alpha = 0.5,
        epsilon = 10e-6
    ):
        q, _ = X.shape
        W = self._W
        B = self._B
        for i in range(epoch):
            start = datetime.now()
            Z = X@W.transpose() + B.transpose()
            P = self.softmax_matrix(Z) 
            accuracy = np.mean(P[np.arange(q), Y])
            loss = -np.mean(np.log(P[np.arange(q), Y] + 1e-12))
            P[np.arange(q), Y] -= 1
            Q = 1/q*P.transpose()*(class_weight.astype('float32'))
            grad_W = Q@X
            grad_B = np.sum(Q, axis=1, keepdims=True)
            if(np.linalg.norm(grad_W) < epsilon and np.linalg.norm(grad_B) < epsilon):
                break
            W -= grad_W*alpha
            B -= grad_B*alpha
            end = datetime.now()
            print(f"epoch {i} {loss} {accuracy} -> {end - start}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = self._W
        B = self._B
        return self.softmax_matrix(X@W.transpose() + B.transpose())
    
    def get_weight(self)->list[list[float]]:
        return self._W.tolist()
    
    def get_bias(self)->list[list[float]]:
        return self._B.tolist()

# a = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# b = np.array([0, 1, 2])

# print(a[np.arange(3), b])