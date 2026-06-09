import pickle
import numpy as np


class KNearestNeighbors:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = None
        self.Y_train: np.ndarray = None
        self.n_classes: int = 0

    # ------------------------------------------------------------------
    # Hamming distance: số vị trí khác nhau giữa 2 vector nhị phân
    # X      : (n_samples, n_features)  — toàn bộ training set
    # x_query: (n_features,)            — 1 sample cần predict
    # trả về : (n_samples,)             — khoảng cách tới từng sample
    # ------------------------------------------------------------------
    @staticmethod
    def hamming_distances(X: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        return np.sum(X != x_query, axis=1)

    # ------------------------------------------------------------------
    # Fit: chỉ cần lưu lại toàn bộ training data (lazy learning)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        self.n_classes = int(Y.max()) + 1

    # ------------------------------------------------------------------
    # Predict 1 sample: trả về class index có xác suất cao nhất
    # ------------------------------------------------------------------
    def predict(self, X: list | np.ndarray) -> int:
        proba = self.predict_proba(X)
        return int(np.argmax(proba))

    # ------------------------------------------------------------------
    # Predict_proba: trả về vector xác suất (n_classes,)
    # Xác suất của mỗi class = số lần xuất hiện trong K láng giềng / K
    # ------------------------------------------------------------------
    def predict_proba(self, X: list | np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Model chưa được fit hoặc load state.")

        x_query = np.array(X, dtype=np.int8)

        distances = KNearestNeighbors.hamming_distances(self.X_train, x_query)

        # Lấy K index có khoảng cách nhỏ nhất
        k_indices = np.argpartition(distances, self.k)[: self.k]
        k_labels = self.Y_train[k_indices]

        # Đếm số lần xuất hiện của mỗi class trong K láng giềng
        proba = np.zeros(self.n_classes, dtype=np.float64)
        for label in k_labels:
            proba[label] += 1.0
        proba /= self.k

        return proba

    # ------------------------------------------------------------------
    # Serialize / deserialize
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
