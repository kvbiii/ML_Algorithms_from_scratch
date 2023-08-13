from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from sklearn.tree import DecisionTreeRegressor

class Ada_Boost_Regressor():
    def __init__(self, n_estimators=50, learning_rate=1.0, loss="linear", random_state=17):
        self.n_estimators = n_estimators
        self.check_n_estimators(n_estimators=self.n_estimators)
        self.learning_rate = learning_rate
        self.check_learning_rate(learning_rate=self.learning_rate)
        self.loss = loss
        self.check_loss(loss=self.loss)
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
        if not loss in(['linear', 'square', 'exponential']):
            raise TypeError('Wrong type of loss. It should be `linear`, `square`, `exponential`.')
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.y_train = self.check_y(y=y)
        self.feature_importances_ = {f"Feature_{i}": 0 for i in range(0, self.X_train.shape[1])}
        self.sample_weights = self.get_initial_sample_weights(X=self.X_train)
        self.sample_weights, self.alphas, self.estimators = self.build_ada_boost(X=self.X_train, y=self.y_train, sample_weights=self.sample_weights)
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
                raise ValueError(f"X has {X.shape[1]} features, but Ada_Boost_Regressor is expecting {self.X_train.shape[1]} features as input.")
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
        self.original_classes = np.unique(y)
        return y
    
    def get_initial_sample_weights(self, X):
        return np.array([1/X.shape[0] for i in range(0, X.shape[0])])
    
    def build_ada_boost(self, X, y, sample_weights):
        alphas = np.array([])
        estimators = []
        for t in range(0, self.n_estimators):
            estimators.append(DecisionTreeRegressor(max_depth=5, random_state=self.random_state))
            estimators[t].fit(X, y, sample_weight=sample_weights)
            predictions = estimators[t].predict(X)
            estimator_error = self.calculate_estimator_error(sample_weights=sample_weights, y=y, predictions=predictions)
            alphas = np.append(alphas, self.calculate_estimator_weight(estimator_error=estimator_error, learning_rate=self.learning_rate))
            sample_weights = self.update_sample_weights(sample_weights=sample_weights, alpha_of_estimator=alphas[t], y=y, predictions=predictions)
            sample_weights = self.normalize_sample_weights(sample_weights=sample_weights)
            self.feature_importances_.update({key: value+value_from_list for (key, value), value_from_list in zip(self.feature_importances_.items(), estimators[t].feature_importances_*alphas[t])})
        return sample_weights, alphas, estimators
    
    def calculate_estimator_error(self, sample_weights, y, predictions):
        error_vector = self.calculate_loss(y=y, predictions=predictions)
        error_vector = self.normalize_error(error_vector=error_vector)
        if(self.loss == "square"):
            error_vector = error_vector**2
        elif(self.loss == "exponential"):
            error_vector = 1-np.exp(-error_vector)
        return np.dot(sample_weights, error_vector)/np.sum(sample_weights)
    
    def calculate_loss(self, y, predictions):
        return np.abs(y-predictions)
    
    def normalize_error(self, error_vector):
        error_max = error_vector.max()
        if error_max != 0:
            error_vector = error_vector/error_max
        return error_vector

    def calculate_estimator_weight(self, estimator_error, learning_rate):
        return learning_rate*1/2*np.log((1-estimator_error)/(estimator_error+1e-7))
    
    def update_sample_weights(self, sample_weights, alpha_of_estimator, y, predictions):
        error_vector = self.calculate_loss(y=y, predictions=predictions)
        error_vector = self.normalize_error(error_vector=error_vector)
        return sample_weights*np.exp(alpha_of_estimator*error_vector)
    
    def normalize_sample_weights(self, sample_weights):
        return sample_weights/np.sum(sample_weights)
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        predictions = np.stack([estimator.predict(X) for estimator in self.estimators], axis=1)
        sorted_idx = np.argsort(predictions, axis=1)
        sorted_weights = self.alphas[sorted_idx]
        cumulate_weights = np.cumsum(sorted_weights, axis=1)
        median_idx = np.argmax(cumulate_weights >= np.max(cumulate_weights)/2, axis=1)
        median_estimators = sorted_idx[np.arange(len(X)), median_idx]
        return predictions[np.arange(len(X)), median_estimators]
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Ada_Boost_Regressor has to be fitted first.')