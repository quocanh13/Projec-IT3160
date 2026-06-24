from .model import KNearestNeighbors

model = KNearestNeighbors(k=1)
model.load_state("./disease_predictor/models/knn/state.pkl")

def predict(X: list[int]) -> list[float]:
    proba = model.predict_proba(X)  
    return proba.tolist()
