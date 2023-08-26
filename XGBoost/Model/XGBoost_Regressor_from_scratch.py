from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class XGBoost_Regressor():
    def __init__(self, n_estimators=100, learning_rate=0.3, reg_lambda=1, gamma=0, max_depth=5, colsample_bytree=1.0, subsample=1, min_child_weight=1, random_state=17):
        self.n_estimators = n_estimators
        self.check_n_estimators(n_estimators=self.n_estimators)
        self.learning_rate = learning_rate
        self.check_learning_rate(learning_rate=self.learning_rate)
        self.reg_lambda = reg_lambda
        self.check_reg_lambda(reg_lambda=self.reg_lambda)
        self.gamma = gamma
        self.check_gamma(gamma=self.gamma)
        self.max_depth = max_depth
        self.check_max_depth(max_depth=self.max_depth)
        self.colsample_bytree = colsample_bytree
        self.check_colsample_bytree(colsample_bytree=self.colsample_bytree)
        self.subsample = subsample
        self.check_subsample(subsample=self.subsample)
        self.min_child_weight = min_child_weight
        self.check_min_child_weight(min_child_weight=self.min_child_weight)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_n_estimators(self, n_estimators):
        if not isinstance(n_estimators, int):
            raise TypeError('Wrong type of n_estimators. It should be int.')
    
    def check_learning_rate(self, learning_rate):
        if not isinstance(learning_rate, float) and not isinstance(learning_rate, int):
            raise TypeError('Wrong type of learning_rate. It should be float or int.')
    
    def check_reg_lambda(self, reg_lambda):
        if not isinstance(reg_lambda, float) and not isinstance(reg_lambda, int):
            raise TypeError('Wrong type of reg_lambda. It should be float or int.')
    
    def check_gamma(self, gamma):
        if not isinstance(gamma, float) and not isinstance(gamma, int):
            raise TypeError('Wrong type of gamma. It should be float or int.')
    
    def check_max_depth(self, max_depth):
        if not isinstance(max_depth, int) and max_depth != None:
            raise TypeError('Wrong type of max_depth. It should be int.')
        if(max_depth == None):
            self.max_depth = 100
    
    def check_colsample_bytree(self, colsample_bytree):
        if not isinstance(colsample_bytree, float) and not isinstance(colsample_bytree, int):
            raise TypeError('Wrong type of colsample_bytree. It should be float or int from range (0.0, 1.0]')
    
    def check_subsample(self, subsample):
        if not isinstance(subsample, float) and not isinstance(subsample, int):
            raise TypeError('Wrong type of subsample. It should be float or int from range (0.0, 1.0]')
    
    def check_min_child_weight(self, min_child_weight):
        if not isinstance(min_child_weight, float) and not isinstance(min_child_weight, int):
            raise TypeError('Wrong type of min_child_weight. It should be float or int.')

    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.y_train = self.check_y(y=y)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.estimators = self.build_xgboost(X=self.X_train, y=self.y_train)
        self.fit_used = True
    
    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if(self.X_train.shape[1] != X.shape[1]):
                raise ValueError(f"X has {X.shape[1]} features, but Gradient_Boosting_Regressor is expecting {self.X_train.shape[1]} features as input.")
        return X
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        return y
    
    def build_xgboost(self, X, y):
        estimators  = np.array([])
        self.base_prediction = self.calculate_optimal_base_prediction(y=y)
        predictions = np.array([self.base_prediction for _ in range(0, len(y))])
        for t in range(0, self.n_estimators):
            Tree = Extreme_Tree_Regressor(X=X, y=y, predictions=predictions, feature_importances_=self.feature_importances_, reg_lambda=self.reg_lambda, gamma=self.gamma, max_depth=self.max_depth, colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight)
            estimators = np.append(estimators, Tree)
            self.feature_importances_ = estimators[t].feature_importances_
            output_values = estimators[t].output_values
            predictions = predictions+self.learning_rate*output_values
        self.normalized_feature_importances = {key: value/np.sum(list(self.feature_importances_.values())) for key, value in self.feature_importances_.items()}
        return estimators

    def calculate_optimal_base_prediction(self, y):
        return 0.5
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        return (self.base_prediction + self.learning_rate*np.sum([estimator.predict(X) for estimator in self.estimators], axis=0))
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Gradient_Boosting_Regressor has to be fitted first.')
        

class Extreme_Tree_Regressor():
    def __init__(self, X, y, predictions, feature_importances_, reg_lambda, gamma, max_depth, colsample_bytree, min_child_weight):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.feature_importances_ = feature_importances_
        self.tree = self.build_extreme_tree(X=X, y=y, predictions=predictions, depth=0, previous_gain=0)
        self.output_values = self.predict(X=X)

    def build_extreme_tree(self, X, y, predictions, depth, previous_gain=0):
        best_split = self.find_best_split(X=X, y=y, predictions=predictions, previous_gain=previous_gain)
        if(best_split["gain"] != 0 and depth < self.max_depth):
            true_branch = self.build_extreme_tree(X=X[best_split["true_rows"], :], y=y[best_split["true_rows"]], predictions=predictions[best_split["true_rows"]], depth=depth+1, previous_gain=best_split["gain"])
            self.calculate_feature_importance(gain=best_split["gain"], feature=best_split["feature"])
            best_split["gain"] = 0
            false_branch = self.build_extreme_tree(X=X[best_split["false_rows"], :], y=y[best_split["false_rows"]], predictions=predictions[best_split["false_rows"]], depth=depth+1, previous_gain=best_split["gain"])
            self.calculate_feature_importance(gain=best_split["gain"], feature=best_split["feature"])
            return Decision_Node(true_branch=true_branch, false_branch=false_branch, feature=best_split["feature"], interpolation_value=best_split["interpolation_value"])
        return Leaf(y=y, predictions=predictions, reg_lambda=self.reg_lambda)

    def find_best_split(self, X, y, predictions, previous_gain):
        best_split = {"gain": 0, "feature": None, "true_rows": None, "false_rows": None, "interpolation_value": None}
        simmilarity_root_node = self.calculate_simmilarity(y_in_leaf=y, predictions_in_leaf=predictions, reg_lambda=self.reg_lambda)
        limited_columns = self.get_limited_columns(X=X)
        for feature in limited_columns:
            interpolation_values = self.find_interpolation_values(X=X[:,feature], predictions=predictions)
            for interpolation_value in interpolation_values:
                true_rows, false_rows = self.partition(X=X, column=feature, interpolation_value=interpolation_value)
                # Skip this split if it doesn't divide the dataset. If it will happen for all the columns then it means that node is a leaf.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                simmilarity_left_node = self.calculate_simmilarity(y_in_leaf=y[true_rows], predictions_in_leaf=predictions[true_rows], reg_lambda=self.reg_lambda)
                simmilarity_right_node = self.calculate_simmilarity(y_in_leaf=y[false_rows], predictions_in_leaf=predictions[false_rows], reg_lambda=self.reg_lambda)
                cover_left = self.calculate_cover(predictions_in_leaf=predictions[true_rows])
                cover_right = self.calculate_cover(predictions_in_leaf=predictions[false_rows])
                gain = self.calculate_gain(root_node_simmilarity=simmilarity_root_node, left_node_simmilarity=simmilarity_left_node, right_node_simmilarity=simmilarity_right_node)
                if gain >= best_split["gain"] and gain > previous_gain and gain > self.gamma and cover_left >= self.min_child_weight and cover_right >= self.min_child_weight:
                    best_split["gain"] = gain
                    best_split["feature"] = feature
                    best_split["true_rows"] = true_rows
                    best_split["false_rows"] = false_rows
                    best_split["interpolation_value"] = interpolation_value
        return best_split

    def get_limited_columns(self, X):
        return random.sample([i for i in range(0, X.shape[1])], int(X.shape[1]*self.colsample_bytree))
    
    def find_interpolation_values(self, X, predictions):
        #return self.weighted_quantile_sketch(X=X, predictions=predictions)
        if(len(np.unique(X)) > 5):
            return set([np.quantile(X, q=i, method="midpoint") for i in [0, 0.2, 0.4, 0.6, 0.8, 1]])
        else:
            return (X[1:] + X[:-1]) / 2
    
    def weighted_quantile_sketch(self, X, predictions):
        split_points = []
        hessian_ = 2*len(predictions)
        df = pd.DataFrame({'feature': X,'hess':hessian_})
        df.sort_values(by=['feature'], ascending = True, inplace = True)
        df['rank'] = df.apply(lambda x : (1/df['hess'].sum())*sum(df[df['feature'] < x['feature']]['hess']), axis=1)
        for row in range(df.shape[0]-1):
            rk_sk_j, rk_sk_j_1 = df['rank'].iloc[row:row+2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if(diff < self.epsilon):
                split_points.append((df["feature"].loc[df["rank"]==rk_sk_j].values[0]+df["feature"].loc[df["rank"]==rk_sk_j_1].values[0])/2)
        return set(split_points)

    def partition(self, X, column, interpolation_value):
        return np.where(np.array(X)[:,column] <= interpolation_value)[0], np.where(np.array(X)[:,column] > interpolation_value)[0]

    def calculate_simmilarity(self, y_in_leaf, predictions_in_leaf, reg_lambda):
        return 1/2*((-2*np.sum(y_in_leaf-predictions_in_leaf))**2)/(2*len(y_in_leaf)+reg_lambda)
    
    def calculate_cover(self, predictions_in_leaf):
        return len(predictions_in_leaf)
    
    def calculate_gain(self, root_node_simmilarity, left_node_simmilarity, right_node_simmilarity):
        return left_node_simmilarity+right_node_simmilarity-root_node_simmilarity
    
    def calculate_feature_importance(self, gain, feature):
        self.feature_importances_[f"Feature_{feature}"] += gain
    
    def predict(self, X):
        predictions = np.array([])
        for row in range(0, X.shape[0]):
            predictions = np.append(predictions, self.return_output_values_for_each_observation(row=X[row,:], node=self.tree, depth=0))
        return predictions
    
    def return_output_values_for_each_observation(self, row, node, depth=0):
        if isinstance(node, Leaf):
            return node.prediction
        else:
            if(row[node.feature] <= node.interpolation_value):
                return self.return_output_values_for_each_observation(row=row, node=node.true_branch, depth=depth+1)
            else:
                return self.return_output_values_for_each_observation(row=row, node=node.false_branch, depth=depth+1)

class Leaf:
    def __init__(self, y, predictions, reg_lambda):
        self.prediction = 2*np.sum(y-predictions)/(2*len(y)+reg_lambda)

class Decision_Node:
    def __init__(self, true_branch, false_branch, feature, interpolation_value):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.feature = feature
        self.interpolation_value = interpolation_value