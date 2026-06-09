import numpy as np

class LogisticRegression:
    def __init__(self, weight: np.ndarray | None = None, bias: float = 0):
        self.weight = weight
        self.bias = bias
        
    def __call__(self, X: np.ndarray):
        W = self.weight
        Z = X@W + self.bias
        return self.sigmoid(Z)
        
    @staticmethod
    def sigmoid(x: np.ndarray):
        return 1/(1 + np.exp(-x))
    

def mse_loss(
    Y_hat: np.ndarray, 
    Y: np.ndarray,
    class_weight: np.ndarray
) -> float:
    m = Y.shape[0]
    
    loss = (Y_hat - Y)**2 * class_weight
    loss = loss.sum()/m
    return loss.item()

def mse_grad(
    X: np.ndarray, 
    Y_hat: np.ndarray, 
    Y: np.ndarray,
    class_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    m, n = X.shape
    Y_hat = Y_hat.reshape((m, 1))
    Y = Y.reshape((m, 1))
    class_weight = class_weight.reshape((m, 1))
    
    delta = 2/m*(Y_hat - Y)*Y_hat*(1 - Y_hat)*class_weight
    
    dhw = X.T @ delta
    dhb = np.sum(delta)
    
    return dhw, dhb

def binary_cross_entropy_loss(
    Y_hat: np.ndarray, 
    Y: np.ndarray,
    class_weight: np.ndarray
) -> float:
    loss = -Y*np.log(Y_hat) - (1 - Y)*np.log(1-Y_hat)
    loss *= class_weight
    return float(np.mean(loss))
    

def binary_cross_entropy_grad(
    X: np.ndarray, 
    Y_hat: np.ndarray, 
    Y: np.ndarray,
    class_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    m, n = X.shape
    Y_hat = Y_hat.reshape((m, 1))
    Y = Y.reshape((m, 1))
    class_weight = class_weight.reshape((m, 1))
    
    delta = (Y_hat - Y)*class_weight/m
    
    dhw = X.T @ delta
    dhb = np.sum(delta)
    
    return dhw, dhb
    
    