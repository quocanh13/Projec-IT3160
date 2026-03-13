import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x: float) -> float:
    return 1/(1 + np.exp(-x))

def d_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s*(1 - s)

def plot():
    x = np.linspace(-20, 20, 200)
    y = d_sigmoid(x)

    plt.plot(x, y)
    plt.title("y = sigmoid(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
plot()