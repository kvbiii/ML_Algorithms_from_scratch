from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from Metrics import *

class CHOW():
    def __init__(self):
        pass
    
    def test(self, fitted_model, feature, data=None):
        model, X, y, features_names, rss_original, target_name = self.read_fitted_model_attributes(fitted_model=fitted_model)
        if(feature in features_names):
            X, index, features_names, feature = self.check_if_feature_is_binary_categorical(df=X, features_names=features_names, feature=feature)
            group_indices, number_of_subsamples = self.find_indices_of_subsamples(df=X, index=index)
            sum_of_rss = self.calculate_sum_of_rss_in_subsamples(model=model, df=X, y=y, group_indices=group_indices, index=index, features_names=features_names, target_name=target_name)
        else:
            self.check_if_feature_is_in_provided_data(feature=feature, data=data)
            data, index, features_names, feature = self.check_if_feature_is_binary_categorical(df=data, features_names=features_names, feature=feature)
            group_indices, number_of_subsamples = self.find_indices_of_subsamples(df=data, index=index)
            sum_of_rss = self.calculate_sum_of_rss_in_subsamples(model=model, df=X, y=y, group_indices=group_indices, index=index, features_names=features_names, target_name=target_name)
        nominator = (rss_original-sum_of_rss)/(len(features_names)*(number_of_subsamples-1))
        denominator = (sum_of_rss/(len(X)-number_of_subsamples*len(features_names)))
        self.F_test = np.round(nominator/denominator, 5)
        self.p_value = np.round(model.calculate_p_value_F_test(F_test=self.F_test, dfn=(number_of_subsamples-1)*len(features_names), dfd=len(X)-number_of_subsamples*len(features_names)), 5)

    def read_fitted_model_attributes(self, fitted_model):
        mod_class = fitted_model.__class__
        model = mod_class(fit_intercept=False, optimization=False, degree=fitted_model.degree)
        X = fitted_model.X.copy()
        y = fitted_model.y.copy()
        features_names = fitted_model.features_names.copy()
        y_pred = fitted_model.predict(X=X, features_names=features_names).copy()
        rss_original =  RSS(y_true=fitted_model.y, y_pred=y_pred)
        return model, X, y, features_names, rss_original, fitted_model.target_name
    
    def check_if_feature_is_in_provided_data(self, feature, data):
        if(feature not in data.columns.tolist()):
            raise ValueError("No feature in the fitted model nor in data provided.")
    
    def check_if_feature_is_binary_categorical(self, df, features_names, feature):
        try:
            index = features_names.index(feature)
            features_names.remove(feature)
        except:
            index = np.where(df.columns==feature)[0][0]
            df = np.array(df)
        if(len(np.unique(df[:,index])) > 15):
            raise ValueError('Provided feature is not categorical nor binary.')
        return df, index, features_names, feature
    
    def find_indices_of_subsamples(self, df, index):
        i = 0
        indices = []
        while(i < len(np.unique(df[:,index]))):
            indices.append(np.where(df[:,index]==np.unique(df[:,index])[i])[0])
            i = i + 1
        return indices, len(indices)

    def calculate_sum_of_rss_in_subsamples(self, model, df, y, group_indices, index, features_names, target_name):
        i = 0
        sum_of_rss = 0
        while(i < len(group_indices)):
            mod_class = model.__class__
            model = mod_class(fit_intercept=False, optimization=False, degree=model.degree)
            globals()[f"X_{i}"] = df[group_indices[i]]
            globals()[f"y_{i}"] = y[group_indices[i]]
            try:
                globals()[f"X_{i}"] = np.delete(globals()[f"X_{i}"], index, axis=1)
            except:
                pass
            model.fit(X=globals()[f"X_{i}"], y=globals()[f"y_{i}"], features_names=features_names, target_name=target_name, diagnostic_test=True)
            y_pred = model.predict(X=globals()[f"X_{i}"], features_names=features_names)
            rss = RSS(y_true=model.y, y_pred=y_pred)
            sum_of_rss = sum_of_rss + rss
            i = i + 1
        return sum_of_rss