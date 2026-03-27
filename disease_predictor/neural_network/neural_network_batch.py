import numpy as np
import random
from datetime import datetime
from get_data import training_data, class_weight

class NeuralNetwork:
    def __init__(
        self,
        layers: list[tuple[int, int, callable, callable]]
    ):
        self._W : list[np.ndarray] = []
        self._B : list[np.ndarray] = []
        self._f : list[callable] = []
        self._f_pr : list[callable] = []
        self._layer_size : list[tuple[int, int]] = []
        self._L0 = len(layers)
        
        for m, n, f, f_pr in layers:
            self._W.append(np.random.randn(m, n) * np.sqrt(1 / n))
            self._B.append(np.zeros((m, 1)))
            self._f.append(f)
            self._f_pr.append(f_pr)
            self._layer_size.append((m, n))
            
    def set_weight(self, weights: list[np.ndarray]):
        if(len(weights) != len(self._W)):
            raise Exception("len(weights) != len(self._W)")
        for i in range(len(weights)):
            if(weights[i].shape == self._W[i].shape):
                self._W[i] = weights[i].copy()
            else:
                raise Exception("weights[i].shape == self._W[i].shape")
    
    def _cal_pre_delta(self):   
        A = self._A
        W = self._W
        B = self._B
        f = self._f
        f_pr = self._f_pr 
        F_pr = self._F_pr
        for L in range(self._L0):
            Z = W[L]@A[L] + B[L]
            A.append(f[L](Z))
            F_pr.append(f_pr[L](Z))
            
    def _cal_delta(self):
        L0 = self._L0
        A = self._A
        W = self._W
        Y = self._Y
        q, = Y.shape
        class_weight = self._class_weight
        F_pr = self._F_pr
        delta = [None]*L0
        A[L0][Y, np.arange(q)] -= 1 
        print(A[L0])
        delta[L0 - 1] = A[L0]*F_pr[L0 - 1]*class_weight
        for L in reversed(range(L0 - 1)):
            delta[L] = (W[L + 1].transpose()@delta[L + 1])*F_pr[L]
        self._delta : list[np.ndarray] = delta
        
    def _train_one(
        self, 
        training_data: tuple[np.ndarray, np.ndarray],
        class_weight: np.ndarray,
        alpha: float = 0.05
    ):
        Y, X = training_data
        self._A = [X]
        self._Y = Y
        self._F_pr = []
        self._class_weight = class_weight
        self._cal_pre_delta()
        self._cal_delta()
        L0 = self._L0
        W = self._W
        B = self._B
        A = self._A
        q, = Y.shape
        delta = self._delta
        for L in range(L0):
            W[L] -= 1/q * delta[L] @ A[L].transpose() * alpha
            B[L] -= 1/q * np.sum(delta[L], axis=1, keepdims=True) * alpha

    def train(
        self, 
        training_data: list[tuple[np.ndarray, np.ndarray]], 
        class_weight: np.ndarray,
        epoch = 20, 
        alpha: float = 0.05
    ):
        """
        Args:
            training_data (list[tuple[np.ndarray, np.ndarray]]): Y(q,) --- X : (n, q) 
            class_weight (np.ndarray): (m, 1)
            epoch (int, optional): Defaults to 20.
            alpha (float, optional): Defaults to 0.05.

        Raises:
            Exception: _description_
        """
        L0 = self._L0
        m, _ = self._layer_size[L0 - 1]
        if((m, 1) != class_weight.shape):
            raise Exception("(m, 1) != class_weight.shape")
        
        for _ in range(epoch):
            start = datetime.now()
            random.shuffle(training_data)
            for data in training_data:
                self._train_one(data, class_weight, alpha)
            end = datetime.now()
            print(end - start)

    def predict(self, X: np.ndarray):
        A = X
        W = self._W
        B = self._B
        f = self._f
        
        for L in range(self._L0):
            Z = W[L]@A + B[L]
            A = f[L](Z)
            
        return A

def sigmoid(X: np.ndarray):
    return 1/(1 + np.exp(-X))
def d_sigmoid(X: np.ndarray):
    s = sigmoid(X)
    return s*(1-s)

def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

nn = NeuralNetwork([
    (20, 377, relu, d_relu),
    (20, 20, relu, d_relu),
    (772, 20, sigmoid, d_sigmoid)
])

nn.train(training_data, class_weight, 2, 0.01)
print(nn.predict(training_data[0][1]))
print(np.argmax(nn.predict(training_data[0][1]), axis=0))
print(training_data[0][0])
# a = np.array([
#     1, 2, 3
# ])
# b = np.zeros((4, 4))
# b[np.arange(0, 3), a] -= 1
# print(np.arange(0, 3))