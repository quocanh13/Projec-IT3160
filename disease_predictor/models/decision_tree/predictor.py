from model import DecisionTree

model = DecisionTree()

def predict(X: list[int]) -> list[float]:
    pred = model.predict(X)
    res = [0.0 for _ in range(773)]
    res[pred] = 1
    return res
    