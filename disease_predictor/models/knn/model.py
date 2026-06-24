import pickle
import numpy as np


class KNearestNeighbors:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray = None
        self.Y_train: np.ndarray = None
        self.n_classes: int = 0

    @staticmethod
    def hamming_distances(X: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        return np.sum(X != x_query, axis=1)


    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        self.n_classes = int(Y.max()) + 1

    def predict(self, X: list | np.ndarray) -> int:
        proba = self.predict_proba(X)
        return int(np.argmax(proba))


    def predict_proba(self, X: list | np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("Model chưa được fit hoặc load state.")

        x_query = np.array(X, dtype=np.int8)

        distances = KNearestNeighbors.hamming_distances(self.X_train, x_query)

        k_indices = np.argpartition(distances, self.k)[: self.k]
        k_labels = self.Y_train[k_indices]

        proba = np.zeros(self.n_classes, dtype=np.float64)
        for label in k_labels:
            proba[label] += 1.0
        proba /= self.k

        return proba

    def save_state(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
