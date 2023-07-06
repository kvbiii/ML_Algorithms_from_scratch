from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from Metrics import *

class VIF():
    def __init__(self,):
        pass

    def test(self, fitted_model):
        X, features_names = self.read_fitted_model_attributes(fitted_model=fitted_model)
        X, features_names = self.check_intercept(fitted_model=fitted_model, X=X, features_names=features_names)
        r_squared = self.find_vif_for_variables(fitted_model=fitted_model, X=X, features_names=features_names)
        self.scores = self.scores_for_each_feature(features_names=features_names, r_squared=r_squared)
    
    def read_fitted_model_attributes(self, fitted_model):
        X = fitted_model.X.copy()
        features_names = fitted_model.features_names.copy()
        return X, features_names

    def check_intercept(self, fitted_model, X, features_names):
        if(fitted_model.fit_intercept == True):
            X = np.delete(X, 0, axis=1)
            features_names.remove("Intercept")
        return X, features_names
    
    def find_vif_for_variables(self, fitted_model, X, features_names):
        i = 0
        r_squared = []
        while(i < len(features_names)):
            X_copy = X.copy()
            features_names_copy = features_names.copy()
            current_target = X_copy[:, i]
            current_target_name = features_names_copy[i]
            X_copy = np.delete(X_copy, i, axis=1)
            features_names_copy.remove(features_names[i])
            mod_class = fitted_model.__class__
            new_model = mod_class(fit_intercept=fitted_model.fit_intercept, optimization=False, degree=fitted_model.degree)
            new_model.fit(X=X_copy, y=current_target, features_names=features_names_copy, target_name=current_target_name, diagnostic_test=True)
            r_squared.append(new_model.R_square)
            i = i + 1
        return r_squared
    
    def scores_for_each_feature(self, features_names, r_squared):
        vif_dict = {}
        tolerance_dict = {}
        i = 0 
        while(i < len(features_names)):
            vif_dict[features_names[i]] = np.round(1/(1-r_squared[i]), 4)
            tolerance_dict[features_names[i]] = np.round(1-r_squared[i], 4)
            i = i + 1
        vif_dict["Average"] = np.round(np.mean(list(vif_dict.values())), 4)
        tolerance_dict["Average"] = np.round(np.mean(list(tolerance_dict.values())), 4)
        df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})
        return df_vif