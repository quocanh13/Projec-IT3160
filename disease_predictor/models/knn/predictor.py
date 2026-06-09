from .model import KNearestNeighbors

model = KNearestNeighbors(k=1)
model.load_state("./disease_predictor/models/knn/state.pkl")

def predict(X: list[int]) -> list[float]:
    """
    Nhận vào vector triệu chứng (376 chiều, nhị phân 0/1).
    Trả về vector xác suất 773 chiều — mỗi phần tử là xác suất
    tương ứng với bệnh đó trong K láng giềng gần nhất.
    """
    proba = model.predict_proba(X)   # ndarray (773,)
    return proba.tolist()
