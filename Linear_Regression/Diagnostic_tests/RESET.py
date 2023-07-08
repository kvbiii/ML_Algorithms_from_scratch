from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *

class RESET():
    def __init__(self):
        pass
    
    def test(self, fitted_model, power=3):
        model, X, y, y_pred_restricted, features_names, rss_restricted, target_name = self.read_fitted_model_attributes(fitted_model=fitted_model)
        power = self.check_power(power=power)
        X, features_names = self.add_predictions_to_model(X=X, y_pred_restricted=y_pred_restricted, features_names=features_names, power=power)
        new_model = self.fit_model(model=model, X=X, y=y, features_names=features_names, target_name=target_name)
        self.F_test, self.p_value = self.calculate_F_test_and_p_value(model=new_model, X=X, y=y, features_names=features_names, rss_restricted=rss_restricted, power=power)
    
    def read_fitted_model_attributes(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=False, optimization=False, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        y = fitted_model.y.copy()
        features_names = fitted_model.features_names.copy()
        y_pred = fitted_model.predict(X=X, features_names=features_names).copy()
        rss_original =  RSS(y_true=fitted_model.y, y_pred=y_pred)
        return model, X, y, y_pred, features_names, rss_original, fitted_model.target_name
    
    def check_power(self, power):
        if(power < 2):
            raise ValueError('Power has to be greater or equal to 2.')
        if(type(power) != int):
            raise TypeError("Power must be int value.")
        return power
    
    def add_predictions_to_model(self, X, y_pred_restricted, features_names, power):
        for p in range(2, power+1):
            X = np.column_stack([X, y_pred_restricted**p])
            features_names.append(f"y_pred^{p}")
        return X, features_names

    def fit_model(self, model, X, y, features_names, target_name):
        model.fit(X=X, y=y, features_names=features_names, target_name=target_name, diagnostic_test=True)
        return model
    
    def calculate_F_test_and_p_value(self, model, X, y, features_names, rss_restricted, power):
        y_pred_unrestricted = model.predict(X=X, features_names=features_names)
        rss_unrestricted = RSS(y_true=y, y_pred=y_pred_unrestricted)
        F_test = np.round(((rss_restricted-rss_unrestricted)/(power-1))/((rss_unrestricted)/(len(X)-len(features_names))), 5)
        p_value = np.round(model.calculate_p_value_F_test(F_test=F_test, dfn=power-1, dfd=len(X)-len(features_names)), 5)
        return F_test, p_value