from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Random_Forest_Classifier():
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, max_features=None, bootstrap=True, class_weight=None, random_state=17):
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
        self.class_weight = class_weight
        self.check_class_weight(class_weight=self.class_weight)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_n_estimators(self, n_estimators):
        if not isinstance(n_estimators, int):
            raise TypeError('Wrong type of n_estimators. It should be int.')
    
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
    
    def check_bootstrap(self, bootstrap):
        if not isinstance(bootstrap, bool):
            raise TypeError('Wrong type of bootstrap. It should be True or False')
    
    def check_class_weight(self, class_weight):
        if not isinstance(class_weight, dict) and class_weight!="balanced" and class_weight!=None:
            raise TypeError("Wrong type of class_weight. It should be dict, string: `balanced` or None.")
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.update_class_weight(y=self.y_train)
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
    
    def get_bootstrap_data(self, X, y, iter):
        if(self.bootstrap == True):
            X_bootstrap = iter[0][0]
            y_bootstrap = iter[1]['y']
            return X_bootstrap, y_bootstrap
        else:
            return X, y
    
    def build_tree(self, X, y, depth):
        information_gain, question, condition, true_rows, false_rows = self.find_best_split(X=X, y=y)
        # Base case: no further info gain. Since we can ask no further questions, we'll return a leaf.
        if information_gain == 0:
            return Leaf(y=y, original_classes=self.original_classes)
        # If we reach here, we have found a useful feature / value to partition on.
        X_true_subset = X[true_rows,:]
        X_false_subset = X[false_rows,:]
        y_true_subset = y[true_rows]
        y_false_subset = y[false_rows]
        # Recursively build the true branch.
        true_branch = self.build_tree(X=X_true_subset, y=y_true_subset, depth=depth+1)
        # Recursively build the false branch.
        false_branch = self.build_tree(X=X_false_subset, y=y_false_subset, depth=depth+1)
        normalized_frequency_of_each_class_in_current_node = self.get_proba(y=y, original_classes=self.original_classes)
        prediction_in_current_node = self.most_frequent_class(y=y)
        if(depth <= self.max_depth):
            self.calculate_node_importance(y=y, true_rows=true_rows, false_rows=false_rows, column=condition["column"])
        return Decision_Node(question, condition, true_branch, false_branch, prediction_in_current_node, normalized_frequency_of_each_class_in_current_node)

    def find_best_split(self, X, y):
        best_information_gain = 0
        best_question = ""
        best_condition = {}
        best_true_rows = None
        best_false_rows = None
        uncertainty_current_subset = self.calculate_uncertainty(y=y)
        condition = {}
        limited_columns = self.get_limited_columns(X=X)
        for column in limited_columns:
            interpolation_values = set([np.quantile(np.array(X)[:,column], q=i, method="midpoint") for i in [0, 0.2, 0.4, 0.6, 0.8, 1]])
            for interpolation_value in interpolation_values:
                question = f"Is feature[{column}] <= {interpolation_value}"
                condition["column"] = column
                condition["interpolation_value"] = interpolation_value
                true_rows, false_rows = self.partition(X=X, column=column, interpolation_value=interpolation_value)
                # Skip this split if it doesn't divide the dataset. If it will happen for all the columns then it means that node is a leaf.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                # Calculate the information gain from this split
                avg_uncertainty_child = self.calculate_avg_uncertainty_child(y=y, true_rows=true_rows, false_rows=false_rows)
                information_gain = self.calculate_information_gain(uncertainty_of_subset=uncertainty_current_subset, avg_uncertainty_child=avg_uncertainty_child)
                if information_gain >= best_information_gain and len(true_rows)+len(false_rows) > self.min_samples_split:
                    best_information_gain, best_question, best_condition, best_true_rows, best_false_rows = information_gain, question, condition.copy(), true_rows, false_rows
        return best_information_gain, best_question, best_condition, best_true_rows, best_false_rows
    
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
    
    def partition(self, X, column, interpolation_value):
        return np.where(np.array(X)[:,column] <= interpolation_value)[0], np.where(np.array(X)[:,column] > interpolation_value)[0]
    
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

    def calculate_information_gain(self, uncertainty_of_subset, avg_uncertainty_child):
        return uncertainty_of_subset - avg_uncertainty_child
    
    def calculate_node_importance(self, y, true_rows, false_rows, column):
        n_node = len(y[true_rows])+len(y[false_rows])
        N = self.X_train.shape[0]
        uncertainty_node = self.calculate_uncertainty(y=y)
        n_subset_left = len(y[true_rows])
        n_subset_right = len(y[false_rows])
        uncertainty_subset_1 = self.calculate_uncertainty(y=y[true_rows])
        uncertainty_subset_2 = self.calculate_uncertainty(y=y[false_rows])
        node_importance = n_node/N*(uncertainty_node-n_subset_left/n_node*uncertainty_subset_1-n_subset_right/n_node*uncertainty_subset_2)
        self.feature_importances_[f"Feature_{column}"] += node_importance
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        predictions = {i: [] for i in range(0, self.n_estimators)}
        for iter in range(0, self.n_estimators):
            for row in range(0, X.shape[0]):
                predictions[iter].append(self.get_score(row=X[row,:], node=self.trees[iter], problem="prediction"))
        return np.array([Counter(col).most_common(1)[0][0] for col in zip(*list(predictions.values()))])
    
    def predict_proba(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        probabilities = {i: [] for i in range(0, self.n_estimators)}
        for iter in range(0, self.n_estimators):
            for row in range(0, X.shape[0]):
                probabilities[iter].append(self.get_score(row=X[row,:], node=self.trees[iter], problem="probability"))
        probabilities_values = np.array(list(probabilities.values()))
        return np.array([list(np.sum(probabilities_values[:,element], axis=0)/len(probabilities_values[:,element])) for element in range(0, probabilities_values.shape[1])])
    
    def get_score(self, row, node, problem, depth=0):
        if isinstance(node, Leaf):
            if(problem == "prediction"):
                return node.prediction
            else:
                return node.probabilities
        elif(depth <= self.max_depth):
            if(row[node.condition["column"]] <= node.condition["interpolation_value"]):
                return self.get_score(row=row, node=node.true_branch, problem=problem, depth=depth+1)
            else:
                return self.get_score(row=row, node=node.false_branch, problem=problem, depth=depth+1)
        else:
            if(problem == "prediction"):
                return node.prediction_in_current_node
            else:
                return node.normalized_frequency_of_each_class_in_current_node
    
    def most_frequent_class(self, y):
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]
    
    def get_proba(self, y, original_classes):
        return [len(np.where(y==original_classes[iter_of_class])[0])/len(y) for iter_of_class in range(0, len(original_classes))]
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Random_Forest_Classifier has to be fitted first.')

    def print_tree(self, node=None, spacing=""):
        if(node == None):
            node = self.tree
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.prediction)
            return
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

class Leaf:
    def __init__(self, y, original_classes):
        self.prediction = Random_Forest_Classifier().most_frequent_class(y=y)
        self.probabilities = Random_Forest_Classifier().get_proba(y=y, original_classes=original_classes)

class Decision_Node:
    def __init__(self, question, condition, true_branch, false_branch, prediction_in_current_node, normalized_frequency_of_each_class_in_current_node):
        self.question = question
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction_in_current_node = prediction_in_current_node
        self.normalized_frequency_of_each_class_in_current_node = normalized_frequency_of_each_class_in_current_node