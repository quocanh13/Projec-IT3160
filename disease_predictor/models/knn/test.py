from disease_predictor.models.knn.model import KNearestNeighbors
from disease_predictor.models.knn.dataset import X_test, Y_test
import numpy as np
# python -m disease_predictor.models.knn.test
model = KNearestNeighbors(k=5)
model.load_state("./disease_predictor/models/knn/state.pkl")

correct = 0
for i in range(X_test.shape[0]):
    res = model.predict(X_test[i])
    if(res == Y_test[i]):
        correct += 1

print(correct / X_test.shape[0])