from disease_predictor.models.knn.model import KNearestNeighbors
from disease_predictor.models.knn.dataset import X_train, Y_train
import numpy as np

model = KNearestNeighbors(k=5)
model.fit(X_train, Y_train)
model.save_state("./disease_predictor/models/knn/state.pkl")

sample = [0] * 377
proba  = model.predict_proba(sample) 

top5 = np.argsort(proba)[::-1][:5]
print("Top-5 bệnh:")
for idx in top5:
    if proba[idx] > 0:
        print(f"  class {idx:>4d}  →  {proba[idx]*100:.1f}%")
