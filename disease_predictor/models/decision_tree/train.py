from model import DecisionTree
from dataset import X, Y

tree = DecisionTree()
# tree.fit(X, Y)
# tree.save_state("./disease_predictor/models/decision_tree/state.pkl")

tree.load_state("./disease_predictor/models/decision_tree/state.pkl")
print(tree.predict([0 for i in range(376)]))