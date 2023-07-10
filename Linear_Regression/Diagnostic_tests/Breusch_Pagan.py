from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *

class Breusch_Pagan():
    def __init__(self):
        pass
    
    def test(self, fitted_model):
        model, X, features_names, residuals_squared = self.read_fitted_model_attributes(fitted_model=fitted_model)
        new_model = self.fit_model_with_residuals(model=model, X=X, residuals=residuals_squared, features_names=features_names, target_name="residuals_squared")
        self.F_test, self.p_value = self.calculate_F_test_and_p_value(model=new_model, X=new_model.X, y=new_model.y, features_names=new_model.features_names)

    def read_fitted_model_attributes(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=False, optimization=False, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        y_pred = fitted_model.predict(X=X, features_names=features_names).copy()
        residuals_squared = (fitted_model.y-y_pred)**2
        return model, X, features_names, residuals_squared
    
    def fit_model_with_residuals(self, model, X, residuals, features_names, target_name):
        model.fit(X=X, y=residuals, features_names=features_names, target_name=target_name, diagnostic_test=True)
        return model

    def calculate_F_test_and_p_value(self, model, X, y, features_names):
        R_squared = R_square(y_true=y, y_pred=model.predict(X=X, features_names=features_names))
        nominator = (R_squared/len(features_names))
        denominaotr = (1-R_squared)/(len(X)-len(features_names)-1)
        F_test = np.round(nominator/denominaotr, 5)
        p_value = np.round(model.calculate_p_value_F_test(F_test=F_test, dfn=len(features_names), dfd=len(X)-len(features_names)-1), 5)
        return F_test, p_value