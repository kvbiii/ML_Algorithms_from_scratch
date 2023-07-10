from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *

class Jarque_Bera():
    def __init__(self):
        pass

    def test(self, fitted_model):
        residuals, features_names = self.read_fitted_model_attributes(fitted_model=fitted_model)
        self.skewness = self.calculate_skewness(residuals=residuals)
        self.kurtosis = self.calculate_kurtosis(residuals=residuals)
        self.Jarque_Bera_Statistic = self.calculate_jarque_bera_statistic(residuals=residuals, skewness=self.skewness, kurtosis=self.kurtosis, number_of_independent_variables_and_const=len(features_names))
        self.p_value = self.calculate_p_value(Jarque_Bera_Statistic=self.Jarque_Bera_Statistic)

    
    def read_fitted_model_attributes(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=fitted_model.fit_intercept, optimization=fitted_model.optimization, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        y_pred_restricted = fitted_model.predict(X=X, features_names=features_names).copy()
        residuals = fitted_model.y-y_pred_restricted
        return residuals, features_names
    
    def calculate_skewness(self, residuals):
        nominator = 1/len(residuals)*np.sum((residuals-np.mean(residuals))**3)
        denominator = (1/len(residuals)*np.sum((residuals-np.mean(residuals))**2))**(3/2)
        return nominator/denominator
    
    def calculate_kurtosis(self, residuals):
        nominator = 1/len(residuals)*np.sum((residuals-np.mean(residuals))**4)
        denominator = (1/len(residuals)*np.sum((residuals-np.mean(residuals))**2))**(2)
        return nominator/denominator
    
    def calculate_jarque_bera_statistic(self, residuals, skewness, kurtosis, number_of_independent_variables_and_const):
        return np.round((len(residuals))/6*(skewness**2+1/4*(kurtosis-3)**2), 5)
    
    def calculate_p_value(self, Jarque_Bera_Statistic):
        return np.round(1-stats.chi2.cdf(Jarque_Bera_Statistic, 2), 5)