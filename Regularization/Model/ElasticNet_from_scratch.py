from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class ElasticNet_Regressor():
    def __init__(self, l1_term=1, l2_term=1, fit_intercept=True, learning_rate=0.01, max_iter=500, random_state=17):
        self.l1_term = l1_term
        self.l2_term = l2_term
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.fit_used = False
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.X_train = self.check_intercept(X=self.X_train)
        self.coef_ = self.initialize_weights()
        self.find_coefficients_gradient_descent(X=self.X_train, y=self.y_train, learning_rate=self.learning_rate, max_iter=self.max_iter)
        self.fit_used = True

    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if((self.fit_intercept == False and self.X_train.shape[1] != X.shape[1]) or (self.fit_intercept == True and self.X_train.shape[1] != X.shape[1]+1)):
                raise ValueError(f"X has {X.shape[1]} features, but ElasticNet is expecting {self.X_train.shape[1]} features as input.")
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        self.original_shape = list(y.shape)
        if(y.ndim == 2):
            y = y.squeeze()
        return y
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)

    def check_intercept(self, X):
        if(self.fit_intercept==True):
            X = np.column_stack([np.ones(shape=(X.shape[0],1)), X])
        return X

    def initialize_weights(self):
        init_coef = np.random.random(size=(self.X_train.shape[1], 1))
        return init_coef
    
    def find_coefficients_gradient_descent(self, X, y, learning_rate, max_iter):
        self.losses = np.zeros(shape=(max_iter, ))
        for epoch in range(0, max_iter):
            y_predict = self.linear_model(X=X, coef=self.coef_)
            grad_coef = self.calculate_derivative_of_loss(X=X, y=y, y_predict=y_predict, l1_term=self.l1_term, l2_term=self.l2_term)
            self.coef_ = self.coef_ - learning_rate*grad_coef
            self.losses[epoch] = self.calculate_loss(X=X, y=y, y_predict=y_predict, l1_term=self.l1_term, l2_term=self.l2_term)
    
    def linear_model(self, X, coef):
        return np.matmul(X, coef).squeeze()
    
    def calculate_derivative_of_loss(self, X, y, y_predict, l1_term, l2_term):
        return (-2/X.shape[0]*(np.matmul(X.T, y-y_predict)-l1_term/2*np.sum(self.coef_/np.abs(self.coef_))-l2_term*np.sum(self.coef_))).reshape(-1,1)

    def calculate_loss(self, X, y, y_predict, l1_term, l2_term):
        return 1/X.shape[0]*(np.sum(y-y_predict)**2+l1_term*np.sum(np.abs(self.coef_))+l2_term*np.sum(self.coef_**2))
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X = self.check_intercept(X=X)
        self.original_shape[0] = X.shape[0]
        y_predict = np.matmul(X, self.coef_).reshape(self.original_shape)
        return y_predict
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('ElasticNet has to be fitted first.')


class ElasticNet_Classifier():
    def __init__(self, l1_term=1, l2_term=1, fit_intercept=True, learning_rate=0.01, max_iter=500, random_state=17):
        self.l1_term = l1_term
        self.l2_term = l2_term
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.fit_used = False
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.X_train = self.check_intercept(X=self.X_train)
        self.coef_ = self.initialize_weights()
        self.find_coefficients_gradient_descent(X=self.X_train, y=self.y_train, learning_rate=self.learning_rate, max_iter=self.max_iter)
        self.fit_used = True

    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if((self.fit_intercept == False and self.X_train.shape[1] != X.shape[1]) or (self.fit_intercept == True and self.X_train.shape[1] != X.shape[1]+1)):
                raise ValueError(f"X has {X.shape[1]} features, but ElasticNet is expecting {self.X_train.shape[1]} features as input.")
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.reshape(-1,1))
        #Transform to have only 1 and -1 classes
        y_encoded = np.where(y_encoded == 1, 1, -1)
        self.original_categories = encoder.categories_[0]
        if(y_encoded.shape[1] == 2):
            y_encoded = y_encoded[:,1]
        return y_encoded
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)

    def check_intercept(self, X):
        if(self.fit_intercept==True):
            X = np.column_stack([np.ones(shape=(X.shape[0],1)), X])
        return X
    
    def initialize_weights(self):
        if(len(self.original_categories) == 2):
            init_coef = np.random.random(size=(self.X_train.shape[1], 1))
        else:
            init_coef = np.random.random(size=(self.X_train.shape[1], self.y_train.shape[1]))
        return init_coef
    
    def find_coefficients_gradient_descent(self, X, y, learning_rate, max_iter):
        if(len(self.original_categories)==2):
            self.losses = np.zeros(shape=(max_iter, ))
            for epoch in range(0, max_iter):
                    y_predict = self.linear_model(X=X, coef=self.coef_)
                    grad_coef = self.calculate_derivative_of_loss(X=X, y=y, y_predict=y_predict, coef=self.coef_, l1_term=self.l1_term, l2_term=self.l2_term)
                    self.coef_ = self.coef_ - learning_rate*grad_coef
                    self.losses[epoch] = self.calculate_loss(X=X, y=y, y_predict=y_predict, coef=self.coef_, l1_term=self.l1_term, l2_term=self.l2_term)
        else:
            self.losses = np.zeros(shape=(max_iter, len(self.original_categories)))
            for klasa in range(0, len(self.original_categories)):
                for epoch in range(0, max_iter):
                    y_predict = self.linear_model(X=X, coef=self.coef_[:,klasa])
                    grad_coef = self.calculate_derivative_of_loss(X=X, y=y[:,klasa], y_predict=y_predict, coef=self.coef_[:,klasa], l1_term=self.l1_term, l2_term=self.l2_term)
                    self.coef_[:,klasa] = self.coef_[:,klasa] - learning_rate*grad_coef.squeeze()
                    self.losses[epoch, klasa] = self.calculate_loss(X=X, y=y[:,klasa], y_predict=y_predict, coef=self.coef_[:,klasa], l1_term=self.l1_term, l2_term=self.l2_term)
            self.losses = np.mean(self.losses, axis=1)
    
    def linear_model(self, X, coef):
        return np.matmul(X, coef).squeeze()
    
    def calculate_derivative_of_loss(self, X, y, y_predict, coef, l1_term, l2_term):
        return (-2/X.shape[0]*(np.matmul(X.T, y-y_predict)-l1_term/2*np.sum(coef/np.abs(coef))-l2_term*np.sum(coef))).reshape(-1,1)

    def calculate_loss(self, X, y, y_predict, coef, l1_term, l2_term):
        return 1/X.shape[0]*(np.sum(y-y_predict)**2+l1_term*np.sum(np.abs(coef))+l2_term*np.sum(coef**2))

    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X = self.check_intercept(X=X)
        predictions_array = self.get_predictions(X=X)
        final_predictions = self.convert_to_original_classes(predictions_array=predictions_array)
        return final_predictions
    
    def predict_proba(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X = self.check_intercept(X=X)
        predictions_array = self.get_predictions(X=X)
        probabilities_array = self.get_proba(predictions_array=predictions_array)
        return probabilities_array
    
    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('ElasticNet has to be fitted first.')
    
    def get_predictions(self, X):
        return np.matmul(X, self.coef_).squeeze()
    
    def get_proba(self, predictions_array):
        if(len(self.original_categories) == 2):
            scaler = MinMaxScaler()
            probabilities_array = scaler.fit_transform(predictions_array.reshape(-1,1))
            probabilities_array = np.column_stack([1-probabilities_array, probabilities_array])
        else:
            scaler = MinMaxScaler()
            probabilities_array = scaler.fit_transform(predictions_array)
        return probabilities_array
    
    def convert_to_original_classes(self, predictions_array):
        if(len(self.original_categories) == 2):
            y_predict = np.where(predictions_array > 0, self.original_categories[1], self.original_categories[0])
        else:
            indices_of_ones = np.argmax(predictions_array, axis=1)
            y_predict = self.original_categories[indices_of_ones]
        return y_predict