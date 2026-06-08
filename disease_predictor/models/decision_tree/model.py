import pickle
import numpy as np

class DecisionTree:
    class Node:
        def __init__(
            self,
            index: int = None,
            left = None,
            right = None,
        ):
            self.index = index
            self.left = left
            self.right = right
            self.res = None
    
    @staticmethod
    def gini(Y: np.ndarray):
        if(Y.shape[0] == 0):
            return 0.0
        _, counts = np.unique(Y, return_counts=True)
        res = counts / Y.shape[0]
        return 1 - np.sum(res**2)
    
    @staticmethod
    def get_impurity(
        X: np.ndarray, 
        Y: np.ndarray,
        index: int
    ):
        left_mask = X[:, index] == 0
        right_mask = X[:, index] == 1
        
        left_gini = DecisionTree.gini(Y[left_mask])
        right_gini = DecisionTree.gini(Y[right_mask])
        
        impurity = (
            left_gini * left_mask.sum() / Y.shape[0] +
            right_gini * right_mask.sum() / Y.shape[0]
        )
        return impurity
    
    @staticmethod
    def split_data(X: np.ndarray, Y: np.ndarray, index: int):
        left_mask = X[:, index] == 0
        right_mask = X[:, index] == 1
        
        left_X = X[left_mask]
        right_X = X[right_mask]
        left_Y = Y[left_mask]
        right_Y = Y[right_mask]
        
        return left_X, left_Y, right_X, right_Y

    @staticmethod
    def best_feature(
        X: np.ndarray, 
        Y: np.ndarray,
    ):
        impurities = []
        for index in range(X.shape[1]):
            impurity = DecisionTree.get_impurity(X, Y, index)
            impurities.append(impurity)
        impurities = np.array(impurities)
        index = np.argmin(impurities)
        return index, impurities[index]
        
    @staticmethod
    def build_tree(X: np.ndarray, Y: np.ndarray):
        node = DecisionTree.Node()
        values_Y, counts_Y = np.unique(Y, return_counts=True)
        values_X, counts_X = np.unique(X, return_counts=True)
        
        
        if(len(values_Y) == 1):
            node.res = values_Y[0]
        else:    
            index, impurity = DecisionTree.best_feature(X, Y)
            if(np.unique(X[:, index]).shape[0] == 1):
                res = np.argmax(counts_Y)
                res = values_Y[res]
                node.res = res
            else:
                node.index = index
                X_left, Y_left, X_right, Y_right = DecisionTree.split_data(X, Y, index)
                node.left = DecisionTree.build_tree(X_left, Y_left)
                node.right = DecisionTree.build_tree(X_right, Y_right)
        return node
    
    @staticmethod
    def search(root: DecisionTree.Node, X: list):
        while root.res is None:
            if(X[root.index] == 1):
                root = root.right
            else:
                root = root.left
        return root.res
            
    
    def fit(
        self,
        X: np.ndarray, 
        Y: np.ndarray
    ):
        self.root = DecisionTree.build_tree(X, Y)
    
    def predict(
        self,
        X: list | np.ndarray
    ):
        if(isinstance(X, np.ndarray)):
            X = X.tolist()
        return DecisionTree.search(self.root, X)
                
    def save_state(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
            
    def load_state(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.__dict__.update(state)

        