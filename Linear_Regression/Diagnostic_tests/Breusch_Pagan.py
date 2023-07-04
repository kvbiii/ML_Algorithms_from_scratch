from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from Metrics import *

class Breusch_Pagan():
    def __init__(self, power=3):
        self.power = power
        if(self.power < 2):
            raise ValueError('Power has to be greater or equal to 2.')
    
    def test(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=fitted_model.fit_intercept, optimization=fitted_model.optimization, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        y_pred_restricted = fitted_model.predict(X=fitted_model.X, features_names=features_names).copy()
        for power in range(2, self.power+1):
            X = np.column_stack([X, y_pred_restricted**power])
            features_names.append(f"y_pred^{power}")
        model.fit_intercept = False
        model.optimization = False
        model.fit(X=X, y=fitted_model.y, features_names=features_names, target_name=fitted_model.target_name)
        y_pred_unrestricted = model.predict(X=X, features_names=features_names)
        rss_restricted = RSS(y_true=fitted_model.y, y_pred=y_pred_restricted)
        rss_unrestricted = RSS(y_true=fitted_model.y, y_pred=y_pred_unrestricted)
        self.F_test = np.round(((rss_restricted-rss_unrestricted)/(self.power-1))/((rss_unrestricted)/(len(X)-len(fitted_model.features_names)-1)), 5)
        self.p_value = np.round(model.calculate_p_value_F_test(F_test=self.F_test, y=fitted_model.y, number_of_independent_variables_and_const=self.power-1), 5)