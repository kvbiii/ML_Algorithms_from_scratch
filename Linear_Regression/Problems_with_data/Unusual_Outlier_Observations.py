from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Unusual_outlier_observations():
    def __init__(self):
        pass

    def test(self, fitted_model):
        X, residuals = self.read_fitted_model_attributes(fitted_model=fitted_model)
        self.leverage = self.calculate_leverage(X=X)
        self.standarized_residuals = self.calculate_standarized_residuals(X=X, residuals=residuals, leverage=self.leverage)
        self.cook_distance = self.calculate_cook_distance(X=X, standarized_residuals=self.standarized_residuals, leverage=self.leverage)
        self.summary_dataframe = self.summary(leverage=self.leverage, standarized_residuals=self.standarized_residuals, cook_distance=self.cook_distance)
        self.indices_of_outliers = self.find_indices_of_outliers(X=X, standarized_residuals=self.standarized_residuals, leverage=self.leverage, cook_distance=self.cook_distance, summary_dataframe=self.summary_dataframe)
    
    def read_fitted_model_attributes(self, fitted_model):
        X = fitted_model.X.copy()
        y_pred = fitted_model.predict(X=X, features_names=fitted_model.features_names).copy()
        residuals = fitted_model.y-y_pred
        return X, residuals

    def calculate_leverage(self, X):
        inversed_multiply_of_X = np.linalg.inv(np.matmul(X.T, X))
        return np.diag(np.dot(np.dot(X, inversed_multiply_of_X), X.T))
    
    def calculate_standarized_residuals(self, X, residuals, leverage):
        Var_e = 1/(len(X)-X.shape[1])*np.sum(residuals**2)
        standarized_residuals = np.diag(residuals/(np.sqrt(Var_e*(1-leverage))))
        return standarized_residuals
    
    def calculate_cook_distance(self, X, standarized_residuals, leverage):
        return standarized_residuals**2/X.shape[1]*leverage/(1-leverage)
    
    def summary(self, leverage, standarized_residuals, cook_distance):
        return pd.DataFrame({'Leverage': leverage, 'Standarized_residuals': standarized_residuals, "Cook_Distance": cook_distance})
    
    def find_indices_of_outliers(self, X, standarized_residuals, leverage, cook_distance, summary_dataframe):
        return summary_dataframe[(leverage>2*X.shape[1]/len(X)) & (np.abs(standarized_residuals) > 2) & (cook_distance > 4/X.shape[0])].index.tolist()