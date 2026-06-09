from disease_predictor.models.knn.model import KNearestNeighbors
from disease_predictor.models.knn.dataset import X, Y   # X: (247212, 377), Y: (247212,)

import numpy as np

# ── Train ──────────────────────────────────────────────────────────────
model = KNearestNeighbors(k=5)
model.fit(X, Y)
model.save_state("./disease_predictor/models/knn/state.pkl")
print("Đã lưu state KNN.")

# ── Test thử ───────────────────────────────────────────────────────────
sample = [0] * 377
proba  = model.predict_proba(sample)  # mảng 773 phần tử

top5 = np.argsort(proba)[::-1][:5]
print("Top-5 bệnh:")
for idx in top5:
    if proba[idx] > 0:
        print(f"  class {idx:>4d}  →  {proba[idx]*100:.1f}%")
