from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Decision_Tree_Classifier():
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, max_features=None, class_weight=None, random_state=17):
        self.criterion = criterion
        self.check_criterion(criterion=self.criterion)
        self.max_depth = max_depth
        self.check_max_depth(max_depth=self.max_depth)
        self.min_samples_split = min_samples_split
        self.check_min_samples_split(min_samples_split=self.min_samples_split)
        self.max_features = max_features
        self.check_max_features(max_features=self.max_features)
        self.class_weight = class_weight
        self.check_class_weight(class_weight=self.class_weight)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_criterion(self, criterion):
        if not criterion in ["gini", "entropy"]:
            raise ValueError('Wrong value for criterion. It should be `gini` or `entropy`.')
    
    def check_max_depth(self, max_depth):
        if not isinstance(max_depth, int) and max_depth != None:
            raise TypeError('Wrong type of max_depth. It should be int.')
        if(max_depth == None):
            self.max_depth = 100
    
    def check_min_samples_split(self, min_samples_split):
        if not isinstance(min_samples_split, int):
            raise TypeError('Wrong type of min_samples_split. It should be int within a range: [2, Number of observations]')
    
    def check_max_features(self, max_features):
        if not isinstance(max_features, int) and max_features!="log2" and max_features!="sqrt" and max_features != None:
            raise TypeError("Wrong type of max_features. It should be int, or string: `log2` or `sqrt`.")
    
    def check_class_weight(self, class_weight):
        if not isinstance(class_weight, dict) and class_weight!="balanced" and class_weight!=None:
            raise TypeError("Wrong type of class_weight. It should be dict, string: `balanced` or None.")
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.update_class_weight(y=self.y_train)
        self.tree = self.build_tree(X=self.X_train, y=self.y_train, depth=0)
        self.normalized_feature_importances = {key: value/np.sum(list(self.feature_importances_.values())) for key, value in self.feature_importances_.items()}
        self.fit_used = True
    
    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if(self.X_train.shape[1] != X.shape[1]):
                raise ValueError(f"X has {X.shape[1]} features, but Decision_Tree_Classifier is expecting {self.X_train.shape[1]} features as input.")
        return X

    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        self.original_classes = np.unique(y)
        return y
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)

    def update_class_weight(self, y):
        if(self.class_weight == None):
            self.class_weight = {klasa: 1.0 for klasa in np.unique(self.y_train)}
        elif(self.class_weight == "balanced"):
            self.class_weight = {klasa: occurrences/len(y) for klasa, occurrences in zip(np.unique(y, return_counts=True)[0][:], np.unique(y, return_counts=True)[1][:])}
    
    def build_tree(self, X, y, depth):
        best_split = self.find_best_split(X=X, y=y)
        if(best_split["gain"] != 0 and depth < self.max_depth):
            true_branch = self.build_tree(X=X[best_split["true_rows"], :], y=y[best_split["true_rows"]], depth=depth+1)
            self.calculate_node_importance(y=y, true_rows=best_split["true_rows"], false_rows=best_split["false_rows"], feature=best_split["feature"])
            false_branch = self.build_tree(X=X[best_split["false_rows"], :], y=y[best_split["false_rows"]], depth=depth+1)
            self.calculate_node_importance(y=y, true_rows=best_split["true_rows"], false_rows=best_split["false_rows"], feature=best_split["feature"])
            return Decision_Node(true_branch=true_branch, false_branch=false_branch, feature=best_split["feature"], interpolation_value=best_split["interpolation_value"])
        return Leaf(y=y, original_classes=self.original_classes)

    def find_best_split(self, X, y):
        best_split = {"gain": 0, "feature": None, "true_rows": None, "false_rows": None, "interpolation_value": None}
        uncertainty_root_node = self.calculate_uncertainty(y=y)
        limited_columns = self.get_limited_columns(X=X)
        for feature in limited_columns:
            interpolation_values = self.find_interpolation_values(X=X[:,feature])
            for interpolation_value in interpolation_values:
                true_rows, false_rows = self.partition(X=X, feature=feature, interpolation_value=interpolation_value)
                # Skip this split if it doesn't divide the dataset. If it will happen for all the columns then it means that node is a leaf.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                # Calculate the information gain from this split
                avg_uncertainty_child = self.calculate_avg_uncertainty_child(y=y, true_rows=true_rows, false_rows=false_rows)
                information_gain = self.calculate_information_gain(uncertainty_root_node=uncertainty_root_node, avg_uncertainty_child=avg_uncertainty_child)
                if information_gain >= best_split["gain"] and len(true_rows)+len(false_rows) > self.min_samples_split:
                    best_split["gain"] = information_gain
                    best_split["feature"] = feature
                    best_split["true_rows"] = true_rows
                    best_split["false_rows"] = false_rows
                    best_split["interpolation_value"] = interpolation_value
        return best_split
    
    def get_limited_columns(self, X):
        upper_bound = {None: random.sample([i for i in range(0, X.shape[1])], int(X.shape[1])),
                        "sqrt": random.sample([i for i in range(0, X.shape[1])], int(X.shape[1]**0.5)),
                        "log2": random.sample([i for i in range(0, X.shape[1])], int(np.log2(X.shape[1]))),
                        }
        try:
            return upper_bound[self.max_features]
        #Except if self.max_features is int.
        except:
            if(self.max_features <= X.shape[1]):
                return random.sample([i for i in range(0, X.shape[1])], self.max_features)
            else:
                raise ValueError("max_features cannot be bigger than number of columns in X.")
    
    def find_interpolation_values(self, X):
        if(len(np.unique(X)) > 5):
            return set([np.quantile(X, q=i, method="midpoint") for i in [0, 0.2, 0.4, 0.6, 0.8, 1]])
        else:
            return (X[1:] + X[:-1]) / 2
    
    def partition(self, X, feature, interpolation_value):
        return np.where(np.array(X)[:,feature] <= interpolation_value)[0], np.where(np.array(X)[:,feature] > interpolation_value)[0]
    
    def calculate_uncertainty(self, y):
        unique_class_counts = np.unique(y, return_counts=True)[1][:]
        uncertainty = {"gini": 1 - np.sum((unique_class_counts/len(y))**2, axis=0),
                        "entropy": np.sum(-unique_class_counts/len(y)*np.log2(unique_class_counts/len(y)), axis=0)}
        return uncertainty[self.criterion]
    
    def calculate_avg_uncertainty_child(self, y, true_rows, false_rows):
        uncertainty_subset_1 = self.calculate_uncertainty(y=y[true_rows])
        uncertainty_subset_2 = self.calculate_uncertainty(y=y[false_rows])
        occurrences_subset_1 = list(np.unique(y[true_rows], return_counts=True)[1][:])
        occurrences_subset_2 = list(np.unique(y[false_rows], return_counts=True)[1][:])
        for klasa in self.original_classes:
            if(klasa not in np.unique(y[true_rows], return_counts=True)[0][:]):
                occurrences_subset_1.insert(int(klasa), 0)
            if(klasa not in np.unique(y[false_rows], return_counts=True)[0][:]):
                occurrences_subset_2.insert(int(klasa), 0)
        weighted_proportion_subset_1 = np.sum([self.class_weight[int(klasa)]*occurrences_subset_1[int(klasa)] for klasa in self.original_classes], axis=0)/(len(true_rows)+len(false_rows))
        weighted_proportion_subset_2 = np.sum([self.class_weight[int(klasa)]*occurrences_subset_2[int(klasa)] for klasa in self.original_classes], axis=0)/(len(true_rows)+len(false_rows))
        return weighted_proportion_subset_1*uncertainty_subset_1+weighted_proportion_subset_2*uncertainty_subset_2

    def calculate_information_gain(self, uncertainty_root_node, avg_uncertainty_child):
        return uncertainty_root_node - avg_uncertainty_child
    
    def calculate_node_importance(self, y, true_rows, false_rows, feature):
        n_node = len(y[true_rows])+len(y[false_rows])
        N = self.X_train.shape[0]
        uncertainty_root_node = self.calculate_uncertainty(y=y)
        n_subset_left = len(y[true_rows])
        n_subset_right = len(y[false_rows])
        uncertainty_subset_1 = self.calculate_uncertainty(y=y[true_rows])
        uncertainty_subset_2 = self.calculate_uncertainty(y=y[false_rows])
        node_importance = n_node/N*(uncertainty_root_node-n_subset_left/n_node*uncertainty_subset_1-n_subset_right/n_node*uncertainty_subset_2)
        self.feature_importances_[f"Feature_{feature}"] += node_importance
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        probabilities = self.get_proba(X=X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        probabilities = self.get_proba(X=X)
        return probabilities
    
    def get_score(self, row, node, depth=0):
        if isinstance(node, Leaf):
            return node.prediction
        else:
            if(row[node.feature] <= node.interpolation_value):
                return self.get_score(row=row, node=node.true_branch, depth=depth+1)
            else:
                return self.get_score(row=row, node=node.false_branch, depth=depth+1)
    
    def get_proba(self, X):
        probabilities = []
        for row in range(0, X.shape[0]):
            probabilities.append(self.get_score(row=X[row, :], node=self.tree, depth=0))
        return np.array(probabilities)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Decision_Tree_Classifier has to be fitted first.')

class Leaf:
    def __init__(self, y, original_classes):
        self.prediction = [len(np.where(y==original_classes[iter_of_class])[0])/len(y) for iter_of_class in range(0, len(original_classes))]

class Decision_Node:
    def __init__(self, true_branch, false_branch, feature, interpolation_value):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.feature = feature
        self.interpolation_value = interpolation_value