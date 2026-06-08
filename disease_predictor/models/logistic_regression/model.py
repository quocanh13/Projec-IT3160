import numpy as np

class LogisticRegression:
    def __init__(self, weight: np.ndarray | None = None, bias: float = 0):
        self.weight = weight
        self.bias = bias
        
    def __call__(self, X: np.ndarray):
        W = self.weight
        return self.sigmoid(X@W + self.bias) 
        
    @staticmethod
    def sigmoid(x: float):
        return 1/(1 + np.exp(-x))

def mse_grad(
    X: np.ndarray, 
    Y_hat: np.ndarray, 
    Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    m, n = X.shape
    Y_hat = Y_hat.reshape((m, 1))
    Y = Y.reshape((m, 1))
    
    delta = 2/m*(Y_hat - Y)*Y_hat*(1 - Y_hat)
    
    dhw = X.T @ delta
    dhb = np.sum(delta)
    
    return dhw, dhb
    
    