import numpy as np
import math

class SoftmaxLogisticRegression:
    @staticmethod
    def softmax(Z: np.ndarray):
        max_z = np.max(Z, axis=1, keepdims=True)
        Z = Z - max_z
        exp_z = np.exp(Z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.W = weight
        self.B = bias
    
    def __call__(self, X: np.ndarray):
        Z = X@self.W.T + self.B
        Y_hat = self.softmax(Z)
        return Y_hat


def softmax_cross_entropy_loss(
    Y_hat: np.ndarray, 
    Y: np.ndarray
) -> np.ndarray:
    loss = -np.sum(Y*np.log(Y_hat), axis=1, keepdims=True)
    loss = np.mean(loss)
    return loss

def sofmax_Cross_entropy_grad(
    X : np.ndarray,
    Y_hat: np.ndarray, 
    Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q, _ = X.shape
    delta = 1/q * (Y_hat - Y)
    dhw = delta.T @ X
    dhb = np.sum(delta, axis=0).T
    
    return dhw, dhb