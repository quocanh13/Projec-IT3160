from .model import DecisionTree
from .dataset import X_train, Y_train
# python -m disease_predictor.models.decision_tree.train
tree = DecisionTree()
tree.fit(X_train, Y_train)
tree.save_state("./disease_predictor/models/decision_tree/state.pkl")