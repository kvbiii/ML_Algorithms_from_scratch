from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Classification_metrics import *

class Logistic_Regression():
    def __init__(self, fit_intercept=True, random_state=17, learning_rate=0.1, max_iter=500):
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_used = False
    
    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.y_train = self.change_to_one_hot_encode(y=self.y_train)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.X_train = self.check_intercept(X=self.X_train)
        self.coef_ = self.initialize_weights()
        self.find_coefficients_gradient_descent(X=self.X_train, y=self.y_train, learning_rate=self.learning_rate, max_iter=self.max_iter)
        self.fit_used = True

    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        if(train == False):
            if((self.fit_intercept == False and self.X_train.shape[1] != X.shape[1]) or (self.fit_intercept == True and self.X_train.shape[1] != X.shape[1]+1)):
                raise ValueError(f"X has {X.shape[1]} features, but Logistic Regression is expecting {self.X_train.shape[1]} features as input.")
        return np.array(X)
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        self.original_classes = np.unique(y)
        self.number_of_classes = len(self.original_classes)
        return y

    def change_to_one_hot_encode(self, y):
        encoder = OneHotEncoder(sparse_output=False)
        y_transformed = encoder.fit_transform(y.reshape(-1,1))
        return y_transformed

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
        return np.random.random(size=(self.X_train.shape[1], self.number_of_classes-1))
    
    def find_coefficients_gradient_descent(self, X, y, learning_rate, max_iter):
        self.losses = np.zeros(shape=(max_iter, self.number_of_classes-1))
        for klasa in range(1, self.number_of_classes):
            current_coef = self.coef_[:, klasa-1]
            for epoch in range(0, max_iter):
                self.losses[epoch, klasa-1] = self.calculate_log_loss(X=X, y=y[:, klasa], coefs=current_coef)
                grad_coef = self.derivative_of_loss(X=X, y=y[:,klasa], coefs=current_coef)
                current_coef = current_coef - learning_rate*grad_coef
            self.coef_[:, klasa-1] = current_coef

    def derivative_of_loss(self, X, y, coefs):
        return -1/X.shape[0]*np.dot(X.T, y-1/(1+np.exp(-np.matmul(coefs, X.T))))
    
    def calculate_log_loss(self, X, y, coefs):
        return -1/X.shape[0]*np.sum(y*np.matmul(coefs, X.T)-np.log(1+np.exp(np.matmul(coefs, X.T))))
    
    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X = self.check_intercept(X=X)
        probabilities_array = self.get_proba(X=X, coef=self.coef_)
        predictions = np.argmax(probabilities_array, axis=1)
        predictions = self.convert_to_original_classes(predictions=predictions)
        return predictions
    
    def predict_proba(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X = self.check_intercept(X=X)
        probabilities_array = self.get_proba(X=X, coef=self.coef_)
        return probabilities_array

    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Logistic Regression has to be fitted first.')
    
    def get_proba(self, X, coef):
        probabilities_array = np.zeros(shape=(X.shape[0], self.number_of_classes))
        sum_of_all_exp_classes = np.sum(np.exp(np.matmul(X, coef)), axis=1)
        for klasa in range(1, self.number_of_classes):
            linear_model = np.matmul(X, coef[:,klasa-1])
            probabilities_array[:, klasa] = np.exp(linear_model)/(1+sum_of_all_exp_classes)
        probabilities_array[:, 0] = 1/(1+sum_of_all_exp_classes)
        return probabilities_array
    
    def convert_to_original_classes(self, predictions):
        return np.array([self.original_classes[klasa] for klasa in predictions])
    
    def summary(self, features_names):
        summary_frame=pd.DataFrame()
        summary_frame["Variables"] = features_names
        summary_frame["Coefficients"] = self.coef_
        return summary_frame