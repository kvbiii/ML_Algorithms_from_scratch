from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Random_Forest_Regressor():
    def __init__(self, n_estimators=100, criterion="squared_error", max_depth=None, min_samples_split=2, max_features=None, bootstrap=True, random_state=17):
        self.n_estimators = n_estimators
        self.check_n_estimators(n_estimators=self.n_estimators)
        self.criterion = criterion
        self.check_criterion(criterion=self.criterion)
        self.max_depth = max_depth
        self.check_max_depth(max_depth=self.max_depth)
        self.min_samples_split = min_samples_split
        self.check_min_samples_split(min_samples_split=self.min_samples_split)
        self.max_features = max_features
        self.check_max_features(max_features=self.max_features)
        self.bootstrap = bootstrap
        self.check_bootstrap(bootstrap=self.bootstrap)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_n_estimators(self, n_estimators):
        if not isinstance(n_estimators, int):
            raise TypeError('Wrong type of n_estimators. It should be int.')
    
    def check_criterion(self, criterion):
        if not criterion in ["squared_error", "absolute_error"]:
            raise ValueError('Wrong value for criterion. It should be `squared_error` or `absolute_error`.')
    
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
    
    def check_bootstrap(self, bootstrap):
        if not isinstance(bootstrap, bool):
            raise TypeError('Wrong type of bootstrap. It should be True or False')
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.trees = []
        bootstrap_instance = MovingBlockBootstrap(7, self.X_train, y=self.y_train, seed=self.random_state)
        for iter in bootstrap_instance.bootstrap(self.n_estimators):
            X_bootstrap, y_bootstrap = self.get_bootstrap_data(X=self.X_train, y=self.y_train, iter=iter)
            self.tree = self.build_tree(X=X_bootstrap, y=y_bootstrap, depth=0)
            self.trees.append(self.tree)
        self.feature_importances_.update({key: value/self.n_estimators for (key, value) in self.feature_importances_.items()})
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
                raise ValueError(f"X has {X.shape[1]} features, but Random_Forest_Regressor is expecting {self.X_train.shape[1]} features as input.")
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
    
    def get_bootstrap_data(self, X, y, iter):
        if(self.bootstrap == True):
            X_bootstrap = iter[0][0]
            y_bootstrap = iter[1]['y']
            return X_bootstrap, y_bootstrap
        else:
            return X, y
    
    def build_tree(self, X, y, depth):
        best_split = self.find_best_split(X=X, y=y)
        if(best_split["reduction"] != 0 and depth < self.max_depth):
            true_branch = self.build_tree(X=X[best_split["true_rows"], :], y=y[best_split["true_rows"]], depth=depth+1)
            self.calculate_node_importance(y=y, true_rows=best_split["true_rows"], false_rows=best_split["false_rows"], feature=best_split["feature"])
            false_branch = self.build_tree(X=X[best_split["false_rows"], :], y=y[best_split["false_rows"]], depth=depth+1)
            self.calculate_node_importance(y=y, true_rows=best_split["true_rows"], false_rows=best_split["false_rows"], feature=best_split["feature"])
            return Decision_Node(true_branch=true_branch, false_branch=false_branch, feature=best_split["feature"], interpolation_value=best_split["interpolation_value"])
        return Leaf(y=y)

    def find_best_split(self, X, y):
        best_split = {"reduction": 0, "feature": None, "true_rows": None, "false_rows": None, "interpolation_value": None}
        error_root_node = self.calculate_error(y=y)
        limited_columns = self.get_limited_columns(X=X)
        for feature in limited_columns:
            interpolation_values = self.find_interpolation_values(X=X[:,feature])
            for interpolation_value in interpolation_values:
                true_rows, false_rows = self.partition(X=X, feature=feature, interpolation_value=interpolation_value)
                # Skip this split if it doesn't divide the dataset. If it will happen for all the columns then it means that node is a leaf.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                # Calculate the reduction from this split
                avg_error_child = self.calculate_avg_error_child(y=y, true_rows=true_rows, false_rows=false_rows)
                reduction = self.calculate_reduction(error_root_node=error_root_node, avg_error_child=avg_error_child)
                if reduction >= best_split["reduction"] and len(true_rows)+len(false_rows) > self.min_samples_split:
                    best_split["reduction"] = reduction
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
    
    def calculate_error(self, y):
        error = {"squared_error": np.mean((y - np.mean(y)) ** 2),
                "absolute_error": np.mean(np.abs(y - np.mean(y)))}
        return error[self.criterion]
    
    def calculate_avg_error_child(self, y, true_rows, false_rows):
        error_subset_1 = self.calculate_error(y=y[true_rows])
        error_subset_2 = self.calculate_error(y=y[false_rows])
        return len(true_rows)/(len(true_rows)+len(false_rows))*error_subset_1+len(false_rows)/(len(true_rows)+len(false_rows))*error_subset_2

    def calculate_reduction(self, error_root_node, avg_error_child):
        return error_root_node - avg_error_child
    
    def calculate_node_importance(self, y, true_rows, false_rows, feature):
        n_node = len(y[true_rows])+len(y[false_rows])
        N = self.X_train.shape[0]
        error_root_node = self.calculate_error(y=y)
        n_subset_left = len(y[true_rows])
        n_subset_right = len(y[false_rows])
        error_subset_1 = self.calculate_error(y=y[true_rows])
        error_subset_2 = self.calculate_error(y=y[false_rows])
        node_importance = n_node/N*(error_root_node-n_subset_left/n_node*error_subset_1-n_subset_right/n_node*error_subset_2)
        self.feature_importances_[f"Feature_{feature}"] += node_importance
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        predictions = {i: [] for i in range(0, self.n_estimators)}
        for iter in range(0, self.n_estimators):
            for row in range(0, X.shape[0]):
                predictions[iter].append(self.get_score(row=X[row,:], node=self.trees[iter]))
        return np.array([np.mean(np.array(list(Counter(col).items()))[:,0]) for col in zip(*list(predictions.values()))])
    
    def get_score(self, row, node, depth=0):
        if isinstance(node, Leaf):
            return node.prediction
        else:
            if(row[node.feature] <= node.interpolation_value):
                return self.get_score(row=row, node=node.true_branch, depth=depth+1)
            else:
                return self.get_score(row=row, node=node.false_branch, depth=depth+1)
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Random_Forest_Regressor has to be fitted first.')

class Leaf:
    def __init__(self, y):
        self.prediction = np.mean(y)

class Decision_Node:
    def __init__(self, true_branch, false_branch, feature, interpolation_value):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.feature = feature
        self.interpolation_value = interpolation_value