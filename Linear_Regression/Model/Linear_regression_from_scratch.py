from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
from Polynomial import *
from Interactions import *

class Linear_Regression():
    def __init__(self, fit_intercept=True, optimization=False, degree=3):
        self.fit_intercept = fit_intercept
        self.optimization = optimization
        self.degree = degree
        self.fit_used = False
    
    def fit(self, X, y, features_names=None, target_name=None, diagnostic_test=False, robust=False):
        self.fit_used = True
        self.diagnostic_test = diagnostic_test
        self.robust = robust
        self.X = self.check_X(X=X)
        self.y = self.check_y(y=y)
        self.features_names, self.target_name = self.check_features_and_target_names(features_names=features_names, target_name=target_name)
        self.X = self.check_for_object_columns(X=self.X)
        self.X = self.polynomial(X=self.X, features_names=self.features_names)
        self.X, self.features_names = self.interactions(X=self.X, features_names=self.features_names)
        self.X, self.features_names = self.check_intercept(X=self.X, features_names=self.features_names)
        self.find_coefficients_basic_approach(X=self.X, y=self.y)
        self.R_square = R_square(y_true=self.y, y_pred=self.predict(X=self.X, features_names=self.features_names))
        self.R_square_Adjusted = R_square_adjusted(y_true=self.y, y_pred=self.predict(X=self.X, features_names=self.features_names), number_of_independent_variables_and_const=len(self.features_names))
        try:
            self.standard_errors = self.variance_estimator_calculation(X=self.X, y_true=self.y, y_pred=self.predict(X=self.X, features_names=self.features_names), number_of_independent_variables_and_const=len(self.features_names))
        except:
            if(self.diagnostic_test == True):
                return 0
            else:
                raise TypeError('There is to much interactions and the matrix is singular.')
        self.confidence_interval_975 = self.confidence_interval_calculation(coefficients=self.coef_, standard_errors=self.standard_errors, P=0.975)
        self.t_tests = self.calculate_t_test(coefficients=self.coef_, standard_errors=self.standard_errors)
        self.p_values = self.calculate_p_value_t_test(y=self.y, t_tests=self.t_tests, number_of_independent_variables_and_const=len(self.features_names))
        self.F_test = self.calculate_F_test(y_true=self.y, y_pred=self.predict(X=self.X, features_names=self.features_names), number_of_independent_variables_and_const=len(self.features_names))
        self.p_value_F_test = self.calculate_p_value_F_test(F_test=self.F_test, dfn=len(self.features_names), dfd=len(self.X)-len(self.features_names))

    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        return y
    
    def check_features_and_target_names(self, features_names, target_name):
        try:
            features_names = self.X.columns.tolist()
            target_name = self.y.columns[0]
        except: 
            if(features_names == None or target_name == None):
                raise ValueError('No feature names provided')
        return features_names, target_name
    
    def check_for_object_columns(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def interactions(self, X, features_names):
        if(self.optimization==True):
            interactions = Create_Interactions()
            X = interactions.fit_transform(X)
            features_names = interactions.get_feature_names_out(input_features_names=features_names)
            X, features_names = interactions.drop_features_that_are_interaction_of_the_same_feature(X=X, features_names=features_names)
        return X, features_names
    
    def polynomial(self, X, features_names):
        if(self.optimization==True):
            polynomial = PolynomialFeatures(degree=self.degree)
            X = polynomial.fit_transform(X)
            features_names = polynomial.get_feature_names_out(features_names)
        return X

    def check_intercept(self, X, features_names):
        if(self.fit_intercept==True):
            X = np.column_stack([np.ones(shape=(X.shape[0],1)), X])
            features_names.insert(0, "Intercept")
        return X, features_names
    
    def find_coefficients_basic_approach(self, X, y):
        try:
            self.coef_ = np.array(np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y)))
        except:
            self.find_coefficients_gradient_descent(X=X, y=y, learning_rate=0.01, num_epochs=10000)
    
    def find_coefficients_gradient_descent(self, X, y, learning_rate, num_epochs, tolerance=1e-6):
        cost_array = np.zeros(num_epochs)
        self.coef_ = np.zeros(shape=(X.shape[1], 1))
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = y_transformed = scaler.fit_transform(y)
        for i in range(0, num_epochs):
            if(i > 0):
                old_error = error.copy()
            cost, error = self.cost_function(X=X, y=y)
            self.coef_ = self.coef_ - learning_rate*1/X.shape[0]*np.matmul(X.T, error)
            cost_array[i] = cost
            if(i > 0):
                if(np.abs(np.sum(old_error)-np.sum(error))<tolerance):
                    break
        return cost_array

    def cost_function(self, X, y):
        error = self.predict(X, features_names=self.features_names)-y
        error = np.array(error, dtype=np.longdouble)
        X = np.array(X, dtype=np.longdouble)
        cost = 1/(2*X.shape[0])*np.matmul(error.T, error)
        return cost, error
    
    def variance_estimator_calculation(self, X, y_true, y_pred, number_of_independent_variables_and_const):
        if(self.robust == False):
            S_E = RSS(y_true=y_true, y_pred=y_pred)/(len(y_true)-number_of_independent_variables_and_const)
            return (S_E*np.linalg.inv(np.matmul(X.T, X))).diagonal()**0.5
        else:
            inversed_multiply_of_X = np.linalg.inv(np.matmul(X.T, X))
            diagonal_residuals = np.diag(((y_true-y_pred)**2).squeeze())
            return (np.dot(np.dot(np.dot(np.dot(inversed_multiply_of_X, X.T), diagonal_residuals), X), inversed_multiply_of_X)).diagonal()**0.5
    
    def confidence_interval_calculation(self, coefficients, standard_errors, P):
        intervals = np.array([coefficients.T-standard_errors*(1+P), coefficients.T+standard_errors*(1+P)])
        intervals = np.squeeze(intervals, axis=1)
        return intervals
    
    def calculate_t_test(self, coefficients, standard_errors):
        return np.squeeze(coefficients.T/standard_errors)

    def calculate_p_value_t_test(self, y, t_tests, number_of_independent_variables_and_const):
        return np.squeeze(2*(1-stats.t.cdf(np.abs(t_tests), len(y)-number_of_independent_variables_and_const)))
    
    def calculate_F_test(self, y_true, y_pred, number_of_independent_variables_and_const):
        return (TSS(y_true=y_true)-RSS(y_true=y_true, y_pred=y_pred))/(number_of_independent_variables_and_const-(1 if self.fit_intercept==True else 0))/(RSS(y_true=y_true, y_pred=y_pred)/(len(y_true)-number_of_independent_variables_and_const))
    
    def calculate_p_value_F_test(self, F_test, dfn, dfd):
        return 1 - stats.f.cdf(F_test, dfn, dfd)
    
    def predict(self, X, features_names=None):
        if self.fit_used == False:
            raise AttributeError('Linear Regression has to be fitted first.')
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should dataframe, numpy array or torch tensor.')
        features_names = features_names
        try:
            features_names = X.columns.tolist()
        except:
            if(features_names == None):
                raise ValueError('No feature names provided')
        X = np.array(X)
        if(self.optimization==True and np.array_equal(self.X, X)==False):
            X = self.polynomial(X=X, features_names=features_names)
            X, _ = self.interactions(X=X, features_names=features_names)
        if(self.fit_intercept==True and len(self.features_names) != X.shape[1]):
            X = np.column_stack([np.ones(shape=(X.shape[0], 1)), X])
        return np.matmul(X, self.coef_)
    
    def summary(self):
        if self.fit_used == False:
            raise AttributeError('Linear Regression has to be fitted first.')
        returned_string = ""
        znaki_rownosci = ""
        myslniki = ""
        for i in range(72+np.max([len(i) for i in self.features_names])): znaki_rownosci += "="
        for i in range(72+np.max([len(i) for i in self.features_names])): myslniki += "-"
        spacje = "     "
        title = "\033[1m" + "Linear Regression estimations".center(90) + "\033[0m"
        returned_string += title + "\n"
        returned_string += znaki_rownosci + "\n"
        returned_string += "Dep. Variable Name: " + ("Not provided" if self.target_name==None else self.target_name).rjust(40-len("Dep. Variable Name: ")) + spacje
        returned_string += "R-squared: " + str(np.round(self.R_square,5)).rjust(40-len("R-squared: ")) + "\n"
        returned_string += "No. Observations: " + str(self.X.shape[0]).rjust(40-len("No. Observations: ")) + spacje
        returned_string += "R-squared Adjusted: " + str(np.round(self.R_square_Adjusted, 5)).rjust(40-len("R-squared Adjusted: ")) + "\n"
        returned_string += "Df Residuals: " + str(self.X.shape[0]-self.X.shape[1]).rjust(40-len("Df Residuals: ")) + spacje
        returned_string += "F Statistic: " + str(np.round(self.F_test, 5)).rjust(40-len("F Statistic: ")) + "\n"
        returned_string += "Df Model: " + str((self.X.shape[1]-(1 if self.fit_intercept == True else 0))).rjust(40-len("Df Model: ")) + spacje
        returned_string += "Prob(F-statistic): " + str(np.round(self.p_value_F_test, 5)).rjust(40-len("Prob(F-statistic): ")) + "\n"
        returned_string += "White's robust covariance used: " + str("True" if self.robust == True else "False").rjust(40-len("White's robust covariance used: ")) + "\n"
        returned_string += znaki_rownosci + "\n"
        returned_string += "coef".rjust(10+np.max([len(i) for i in self.features_names])) + spacje + "std error".rjust(10) + "t".rjust(10) + spacje + "Prob(t)" + spacje + "[0.025]" + spacje + "[0.975]".rjust(15-len("[0.975]")) + "\n"
        returned_string += myslniki + "\n"
        for i in range(0, len(self.features_names)):
            returned_string += self.features_names[i] + str(np.round(self.coef_[i][0], 3)).rjust(10+np.max([len(i) for i in self.features_names])-len(self.features_names[i])) + str(np.round(self.standard_errors[i], 3)).rjust(15) + str(np.round(self.t_tests[i], 3)).rjust(10) + str(np.round(self.p_values[i], 3)).rjust(12) + str(np.round(self.confidence_interval_975[0][i], 3)).rjust(12) + str(np.round(self.confidence_interval_975[1][i], 3)).rjust(13) + "\n"
        returned_string += znaki_rownosci + "\n"
        return returned_string