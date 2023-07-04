from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from Metrics import *

class White():
    def __init__(self):
        pass
    
    def test(self, fitted_model, interaction_columns_used_in_model):
        if not isinstance(interaction_columns_used_in_model, np.ndarray) and not isinstance(interaction_columns_used_in_model, list):
            raise TypeError("Wrong type, it should be list or numpy array.")
        self.interaction_columns_used_in_model = interaction_columns_used_in_model
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        y_pred_restricted = fitted_model.predict(X=X, features_names=features_names).copy()
        residuals_squared = (fitted_model.y-y_pred_restricted)**2
        mod_class = fitted_model.__class__
        if(fitted_model.fit_intercept == True):
            X = np.delete(X, 0, 1)
            features_names.remove("Intercept")
        i = 0
        while(i < len(self.interaction_columns_used_in_model)):
            index = features_names.index(self.interaction_columns_used_in_model[i])
            features_names.remove(self.interaction_columns_used_in_model[i])
            X = np.delete(X, index, axis=1)
            i = i + 1
        new_model = mod_class(fit_intercept=fitted_model.fit_intercept, optimization=True, degree=2)
        new_model.fit(X=X, y=residuals_squared, features_names=features_names, target_name="residuals_squared", diagnostic_test=True)
        self.F_test = new_model.calculate_F_test(y_true=new_model.y, y_pred=new_model.predict(X=new_model.X, features_names=new_model.features_names), number_of_independent_variables_and_const=len(new_model.features_names))
        self.p_value = np.round(new_model.calculate_p_value_F_test(F_test=self.F_test, dfn=len(features_names), dfd=len(X)-len(features_names)), 5)