import numpy as np
from datetime import datetime
from data.get_data import get_weight
from data.data_creation import mean, std

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

    def __init__(
        self, 
        m: int, 
        n: int, 
        mean: np.ndarray, 
        std: np.ndarray, 
        weight: np.ndarray | None = None, 
        bias: np.ndarray | None = None
    ):
        if(weight is None):
            self._W = np.zeros((m, n))
        else:
            a, b = weight.shape
            if(a != m or b != n):
                print(a, b, m, n)
                raise Exception("Kích thước của weight không đúng")
            else:
                self._W = weight.copy()
        
        if(bias is None):
            self._B = np.zeros((m, 1))
        else:
            a, b = bias.shape
            if(a != m or b != 1):
                raise Exception("Kích thước của bias không đúng")
            else:
                self._B = bias.copy()
        
        self._first_train = True
        self._m = m
        self._n = n
        self._mean = mean.copy()
        self._std = std.copy()

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
        mean = self._mean
        std = self._std
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
        X = (X - mean) / (std + 1e-8) 
        
        pre_loss = 1e9
        for i in range(epoch):
            start_time = datetime.now()
            
            Z = X@W.T + B.T
            P = self.softmax_matrix(Z)
            loss = -(C.flatten() * np.log(P[np.arange(q), Y] + 1e-12)).mean()
            pred = np.argmax(P, axis=1)
            accuracy = np.mean(pred == Y)
            P[np.arange(q), Y] -= 1
            P *= C / q
            grad_W = P.T@X
            grad_B = (np.sum(P, axis=0, keepdims=True)).T
            
            self._W -= alpha*grad_W
            self._B -= alpha*grad_B
            print("Max grad_W", np.max(grad_W))
            end_time = datetime.now()
            
            if(print_res):
                print(f"epoch {i} --> Loss : {loss} - Accuracy : {accuracy} - Time : {end_time - start_time} --> {"good" if loss < pre_loss   else "bad" }")
            pre_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = (X - self._mean) / (self._std + 1e-8)
        Z = X @ self._W.T + self._B.T
        res = self.softmax_matrix(Z)
        return res

    def get_weight(self)->list[list[float]]:
        return self._W.tolist()
    
    def get_bias(self)->list[list[float]]:
        return self._B.tolist()


param = get_weight("softmax_new")
sft = SoftmaxLogisticRegression(
    772, 
    377,
    np.array(mean),
    np.array(std),
    np.array(param["weight"]), 
    np.array(param["bias"])
)

def predict(symptom_list : list[int]) -> list[float]:
    res = sft.predict(np.array(symptom_list))
    return res.tolist()[0]