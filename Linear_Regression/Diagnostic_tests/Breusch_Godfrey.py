from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *

class Breusch_Godfrey():
    def __init__(self):
        pass
    
    def test(self, fitted_model, nlags):
        model, X, features_names, residuals = self.read_fitted_model_attributes(fitted_model=fitted_model)
        nlags = self.check_nlags(nlags=nlags, X=X)
        residuals_with_lags, features_names = self.create_lags_for_residuals(residuals=residuals, nlags=nlags, features_names=features_names)
        X = self.concatenate_with_X(X=X, residuals_with_lags=residuals_with_lags)
        new_model = self.fit_model_with_residuals(model=model, X=X, residuals=residuals, features_names=features_names, target_name="Residuals")
        self.F_test, self.p_value = self.calculate_F_test_and_p_value(model=new_model, X=X, residuals=residuals, features_names=features_names, fitted_model=fitted_model, nlags=nlags)

    def read_fitted_model_attributes(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=False, optimization=False, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        y_pred_restricted = fitted_model.predict(X=X, features_names=features_names).copy()
        residuals = fitted_model.y-y_pred_restricted
        return model, X, features_names, residuals
    
    def check_nlags(self, nlags, X):
        if nlags == None:
            nlags = min(10, len(X)//5)
        else:
            if(type(nlags) != int):
                raise TypeError("nlags must be int value.")
        return nlags
    
    def create_lags_for_residuals(self, residuals, nlags, features_names):
        if(residuals.ndim == 2):
            residuals = residuals.squeeze()
        residuals_with_lags = np.zeros(shape=(residuals.shape[0], nlags))
        flipped_residuals = residuals[::-1]
        i = 0
        while(i < residuals_with_lags.shape[0]):
            if(i > nlags):
                residuals_with_lags[i] = flipped_residuals[-i:-i+nlags]
            elif(i > 0):
                residuals_with_lags[i][0:i] = flipped_residuals[-i:None]
            else:
                pass
            i = i + 1
        residuals_with_lags = np.column_stack([np.ones(shape=(residuals_with_lags.shape[0], 1)), residuals_with_lags])
        i = 0
        while(i < nlags):
            features_names.append(f"e_{i}")
            i = i + 1
        return residuals_with_lags, features_names
    
    def concatenate_with_X(self, X, residuals_with_lags):
        X = np.column_stack([X, residuals_with_lags])
        return X

    def fit_model_with_residuals(self, model, X, residuals, features_names, target_name):
        model.fit(X=X, y=residuals, features_names=features_names, target_name=target_name, diagnostic_test=True)
        return model

    def calculate_F_test_and_p_value(self, model, X, residuals, features_names, fitted_model, nlags):
        R_squared = R_square(y_true=residuals, y_pred=model.predict(X=X, features_names=features_names))
        nominator = (R_squared/nlags)
        denominaotr = (1-R_squared)/(len(X)-nlags-len(fitted_model.features_names))
        F_test = np.round(nominator/denominaotr, 5)
        p_value = np.round(model.calculate_p_value_F_test(F_test=F_test, dfn=nlags, dfd=len(X)-nlags-len(features_names)), 5)
        return F_test, p_value