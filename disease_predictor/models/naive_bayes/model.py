import numpy as np
import pickle
import os

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None
        self.neg_feature_log_prob = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_log_prior[idx] = np.log(X_c.shape[0] / n_samples)
            smoothed_fc = np.sum(X_c, axis=0) + self.alpha
            smoothed_cc = X_c.shape[0] + 2 * self.alpha
            self.feature_log_prob[idx, :] = np.log(smoothed_fc / smoothed_cc)
        self.neg_feature_log_prob = np.log(1 - np.exp(self.feature_log_prob))

    def predict_id(self, X):
        term1 = np.dot(X, self.feature_log_prob.T)
        term2 = np.dot(1 - X, self.neg_feature_log_prob.T)
        posterior_log_prob = self.class_log_prior + term1 + term2
        return self.classes[np.argmax(posterior_log_prob, axis=1)]

    def predict_proba(self, X, temperature=15.0):
        term1 = np.dot(X, self.feature_log_prob.T)
        term2 = np.dot(1 - X, self.neg_feature_log_prob.T)
        posterior_log_prob = self.class_log_prior + term1 + term2
        
        posterior_log_prob = posterior_log_prob / temperature
        max_log_prob = np.max(posterior_log_prob, axis=1, keepdims=True)
        exp_prob = np.exp(posterior_log_prob - max_log_prob)
        probs = exp_prob / np.sum(exp_prob, axis=1, keepdims=True)
        return probs

# --- BÙA CHÚ CHỐNG LỖI KHI LOAD PKL ---
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'NaiveBayes':
            return NaiveBayes
        return super().find_class(module, name)

# --- LOAD TRỌNG SỐ TỪ FILE PKL ---
nb = None
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'naive_bayes_model.pkl')

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as file:
            ai_package = CustomUnpickler(file).load()
            if isinstance(ai_package, dict) and 'model' in ai_package:
                nb = ai_package['model']
            else:
                nb = ai_package
            print("✅ Đã load thành công file naive_bayes_model.pkl")
    except Exception as e:
        print(f"Lỗi load pkl: {e}")

# --- XUẤT API ---
def predict(symptom_list: list[int]) -> list[float]:
    if nb is None:
        return [0.0] * 772
    res = nb.predict_proba(np.array(symptom_list).reshape(1, -1))
    return res.tolist()[0]