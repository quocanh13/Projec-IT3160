import numpy as np

class LogisticRegression:
    def __init__(self, weight: np.ndarray = np.array([]), bias: float = 0):
        self._W: np.ndarray = weight
        self._b = bias
    
    @staticmethod
    def sigmoid(x: float):
        return 1/(1 + np.exp(-x))
    
    def train(self, X: np.ndarray, Y: np.ndarray, weight_class = 1.0, epoch = 100, e = 1e-6, alpha = 0.05):
        m, n = X.shape
        Y = Y.reshape(-1, 1)
        W = self._W
        err = np.ones((m,1))
        err[Y == 1] = weight_class

        for i in range(epoch):
            t: np.ndarray = err*(self.sigmoid(X@W + self._b) - Y)
            grad: np.ndarray = 1/m*(X.transpose()@t)
            # print(grad)
            if(np.linalg.norm(grad) < e): 
                break
            W -= alpha*grad
            self._b -= alpha*t.mean()
            
    def predict(self, X: np.ndarray) -> float:
        W = self._W
        return self.sigmoid(X@W + self._b) 
    
    def print_weight(self):
        print("[")
        for row in self._W:
            for col in row:
                print(f"    [{col}],")
        print("]")
    
    def get_weight(self) -> list[list[float]]:
        return self._W.tolist()  
    def get_bias(self) -> list[list[float]]:
        return self._b
