import numpy as np
import random
import json
from datetime import datetime
# from neural_network.get_data import test_data, training_data

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
        self._W_size = ""
        
        for m, n, f, f_pr in layers:
            self._W.append(np.random.randn(m, n) * np.sqrt(2.0 / n))
            self._B.append(np.zeros((m, 1)))
            self._f.append(f)
            self._f_pr.append(f_pr)
            self._layer_size.append((m, n))
            self._W_size += f"({m}, {n}) "
            
    def set_weight(self, weights: list[np.ndarray]):
        if(len(weights) != len(self._W)):
            raise Exception("len(weights) != len(self._W)")
        for i in range(len(weights)):
            if(weights[i].shape == self._W[i].shape):
                self._W[i] = weights[i].copy()
            else:
                raise Exception("weights[i].shape == self._W[i].shape")
    
    def set_bias(self, bias: list[np.ndarray]):
        if(len(bias) != len(self._B)):
            raise Exception("len(bias) != len(self._B)")
        else:
            for i in range(len(bias)):
                if(bias[i].shape == self._B[i].shape):
                    self._B[i] = bias[i].copy()
                else:
                    raise Exception(f"bias[{i}].shape == self._B[{i}].shape")
    
    def load_weight(self, path = "./neural_network/weight.json"):
        with open(path, "r") as file:
            f : dict = json.load(file)
            size = self._W_size
            p = f.get(size, None)
            if(p != None):
                self.set_weight([np.array(w) for w in p["weight"]])
                self.set_bias([np.array(b) for b in p["bias"]])
                print("Loaded weight")
    
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
            # print(f"Z[{L}] = {Z}")
            F_pr.append(f_pr[L](Z) if f_pr[L] != None else 1)
              
    def _cal_delta(self):
        L0 = self._L0
        A = self._A
        W = self._W
        Y = self._Y
        q, = Y.shape
        class_weight = self._class_weight
        F_pr = self._F_pr
        delta = [None]*L0
        delta[L0-1] = A[L0].copy()*class_weight[Y]
        delta[L0-1][Y, np.arange(q)] -= class_weight[Y]
        for L in reversed(range(L0 - 1)):
            delta[L] = (W[L + 1].transpose()@delta[L + 1])*F_pr[L]
            # print(f"Delta[{L}] = {delta[L]}")
        self._delta : list[np.ndarray] = delta
        
    def _train_one(
        self, 
        training_data: tuple[np.ndarray, np.ndarray],
        class_weight: np.ndarray,
        alpha: float = 0.05,
        r = 0.1
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
            grad_W = (delta[L] @ A[L].T)
            grad_B = np.sum(delta[L], axis=1, keepdims=True)
            # print(f"Max grad_W[{L}] = {np.max(grad_W)}")
            W[L] -= alpha * grad_W / q 
            B[L] -= alpha * grad_B / q 

    def train(
        self, 
        training_data: list[tuple[np.ndarray, np.ndarray]], 
        class_weight: np.ndarray,
        epoch = 20, 
        alpha: float = 0.05,
        r = 0.1,
        write_weight = False
    ):
        """
        Args:
            training_data (list[tuple[np.ndarray, np.ndarray]]): Y(q,) --- X : (n, q) 
            class_weight (np.ndarray): (m, )
            epoch (int, optional): Defaults to 20.
            alpha (float, optional): Defaults to 0.05.

        Raises:
            Exception: _description_
        """
        L0 = self._L0
        m, _ = self._layer_size[L0 - 1]
        if((m, ) != class_weight.shape):
            raise Exception("(m, ) != class_weight.shape")
        
        for i in range(epoch):
            start = datetime.now()
            random.shuffle(training_data)
            for data in training_data:
                self._train_one(data, class_weight, alpha)
            output = self._A[self._L0]
            _, q = output.shape
            eps = 1e-15
            # loss = -np.mean(np.log(np.clip(output[self._Y, np.arange(q)], eps, 1)))
            # end = datetime.now()
            # res = self.predict(test_data[0][1])
            # accuracy = np.mean(np.argmax(res, axis=0) == test_data[0][0])
            
            # print(f"Epoch : {i} -- Time : {end - start} -- Loss : {loss} --- Accuracy : {accuracy}", end="")
            if(write_weight):
                with open("./neural_network/weight.json", "r+") as file:
                    p = json.load(file)
                    file.seek(0)
                    p[self._W_size] = {
                        "weight" : self.get_weight_list(),
                        "bias" : self.get_bias_list()
                    }
                    print(" --- Writing", end="")
                    json.dump(p, file, indent=4)
                    file.truncate()
                    print(" --- Wrote")
            else: print("")

    def predict(self, X: np.ndarray):
        A = X
        W = self._W
        B = self._B
        f = self._f
        
        for L in range(self._L0):
            Z = W[L]@A + B[L]
            A = f[L](Z)
            
        return A
    
    def get_weight_list(self):
        return [w.tolist() for w in self._W]
    def get_bias_list(self):
        return [b.tolist() for b in self._B]

def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

nn = NeuralNetwork(
    [
        (300, 377, relu, d_relu),
        (200, 300, relu, d_relu),
        (772, 200, softmax, None)
    ]
)
nn.load_weight()

def predict(symptom_list : list[int]) -> list[float]:
    res = nn.predict(np.array(symptom_list).reshape((377, 1)))
    return res.tolist()

# nn.train(training_data, class_weight, 10, 0.1, write_weight=False)

# res = nn.predict(test_data[0][1])
# print(np.mean(np.argmax(res, axis=0) == test_data[0][0]))
