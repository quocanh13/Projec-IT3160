from .model import DecisionTree
from .dataset import X_test, Y_test
# python -m disease_predictor.models.decision_tree.test
model = DecisionTree()

model.load_state("./disease_predictor/models/decision_tree/state.pkl")

correct = 0
for i in range(X_test.shape[0]):
    res = model.predict(X_test[i])
    if(res == Y_test[i]):
        print(res, Y_test[i])
        correct += 1
        
print(correct / X_test.shape[0])