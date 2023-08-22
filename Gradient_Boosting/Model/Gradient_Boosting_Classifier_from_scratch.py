from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from sklearn.tree import DecisionTreeRegressor

class Gradient_Boosting_Classifier():
    def __init__(self, n_estimators=50, learning_rate=0.1, loss="log_loss", criterion="squared_error", max_depth=3, random_state=17):
        self.n_estimators = n_estimators
        self.check_n_estimators(n_estimators=self.n_estimators)
        self.learning_rate = learning_rate
        self.check_learning_rate(learning_rate=self.learning_rate)
        self.loss = loss
        self.check_loss(loss=self.loss)
        self.criterion = criterion
        self.check_criterion(criterion=self.criterion)
        self.max_depth = max_depth
        self.check_max_depth(max_depth=self.max_depth)
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.fit_used = False
    
    def check_n_estimators(self, n_estimators):
        if not isinstance(n_estimators, int):
            raise TypeError('Wrong type of n_estimators. It should be int.')
    
    def check_learning_rate(self, learning_rate):
        if not isinstance(learning_rate, float):
            raise TypeError('Wrong type of learning_rate. It should be float.')
    
    def check_loss(self, loss):
        if not loss in ["log_loss"]:
            raise ValueError('Wrong value for loss. It should be `log_loss`')
    
    def check_criterion(self, criterion):
        if not criterion in ["squared_error"]:
            raise ValueError('Wrong value for criterion. It should be `squared_error`.')
    
    def check_max_depth(self, max_depth):
        if not isinstance(max_depth, int) and max_depth != None:
            raise TypeError('Wrong type of max_depth. It should be int.')
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.y_train = self.check_y(y=y)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.estimators = self.build_gradient_boosting(X=self.X_train, y=self.y_train)
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
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor with int or float inputs.')
        y = np.array(y, dtype=np.int64)
        if(y.ndim == 2):
            y = y.squeeze()
        self.original_classes = np.unique(y)
        return y

    def change_to_one_hot_encode(self, y):
        y_transformed = np.zeros((y.size, y.max()+1))
        y_transformed[np.arange(y.size), y] = 1
        return y_transformed

    def build_gradient_boosting(self, X, y):
        all_estimators = []
        self.base_predictions = []
        y = self.change_to_one_hot_encode(y=y)
        for klasa in range(1, len(self.original_classes)):
            residuals = np.zeros(shape=(self.n_estimators, X.shape[0]))
            estimators = np.array([])
            self.base_predictions.append(self.calculate_optimal_base_prediction(y=y[:,klasa]))
            predictions = np.array([self.base_predictions[klasa-1] for _ in range(0, len(y[:,klasa]))])
            for t in range(0, self.n_estimators):
                estimators = np.append(estimators, DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth, random_state=self.random_state))
                if(t > 0):
                    residuals[t] = self.calculate_negative_loss_gradient(y=y[:,klasa], predictions=predictions)
                else:
                    residuals[t] = y[:,klasa]-predictions
                estimators[t].fit(X, residuals[t])
                self.update_leaf_nodes(estimator=estimators[t], X=X, y=y[:,klasa], predictions=predictions)
                predictions = predictions+self.learning_rate*estimators[t].predict(X)
                self.feature_importances_.update({key: value+value_from_list for (key, value), value_from_list in zip(self.feature_importances_.items(), estimators[t].feature_importances_)})
            all_estimators.append(estimators)
        return all_estimators
    
    def calculate_optimal_base_prediction(self, y):
        return -np.log((len(y)-np.sum(y))/np.sum(y))

    def calculate_negative_loss_gradient(self, y, predictions):
        predictions = np.array([100 if i > 100 else -100 if i < -100 else i for i in predictions])
        negative_loss_gradient_dict={"log_loss": lambda y_true, y_pred: (y_true - 1/(1+np.exp(-y_pred)))}
        return negative_loss_gradient_dict[self.loss](y, predictions)
    
    def update_leaf_nodes(self, estimator, X, y, predictions):
        leaf_nodes = np.nonzero(estimator.tree_.children_left == -1)[0]
        node_of_sample = estimator.apply(X)
        for leaf in leaf_nodes:
            samples_in_leaf = np.where(node_of_sample == leaf)[0]
            y_in_leaf = y[samples_in_leaf]
            preds_in_leaf = predictions[samples_in_leaf]
            preds_in_leaf = np.array([100 if i > 100 else -100 if i < -100 else i for i in preds_in_leaf])
            nominator = np.sum(y_in_leaf-1/(1+np.exp(-preds_in_leaf)))
            denominator = np.sum(1/(1+np.exp(-preds_in_leaf))*1/(1+np.exp(preds_in_leaf)))
            gamma = nominator/denominator
            estimator.tree_.value[leaf, 0, 0] = gamma
    
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
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Gradient_Boosting_Classifier has to be fitted first.')
    
    def get_proba(self, X):
        all_log_of_odds = np.array([(self.base_predictions[klasa-1] + self.learning_rate*np.sum([estimator.predict(X) for estimator in self.estimators[klasa-1]], axis=0)) for klasa in range(1, len(self.original_classes))])
        exp_array = np.array([np.exp(log_of_odds) for log_of_odds in all_log_of_odds])
        sum_of_all_exp = np.sum(exp_array, axis=0)
        probabilities_array = np.zeros(shape=(X.shape[0], len(self.original_classes)))
        for klasa in range(1, len(self.original_classes)):
            log_of_odds = (self.base_predictions[klasa-1] + self.learning_rate*np.sum([estimator.predict(X) for estimator in self.estimators[klasa-1]], axis=0))
            probabilities_array[:, klasa] = np.exp(log_of_odds)/(1+sum_of_all_exp)
        probabilities_array[:, 0] = 1/(1+sum_of_all_exp)
        return probabilities_array