from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Features_optimization():
    def __init__(self, cross_validation_instance=KFold(n_splits=5, shuffle=True, random_state=17)):
        self.cross_validation_instance = cross_validation_instance

    def compare_features(self, df, metric, algorithm_instance, feature_set, target_name, compare=True, verbose=False):
        df = self.check_df(df=df)
        features_names = df.columns.tolist()
        features_names.remove(target_name)
        feature_set = self.check_feature_set(feature_set=feature_set, features_names=features_names)
        train_scores, valid_scores = [], []
        if(compare==True):
            for feature in feature_set:
                train_score, valid_score = self.cross_validation(X=np.array(df[[feature]]), y=np.array(df[[target_name]]).squeeze(), metric=metric, algorithm_instance=algorithm_instance, cross_validation_instance=self.cross_validation_instance)
                if(verbose == True):
                    print("Feature: {}; Mean of Train Scores: {}; Mean of Valid Scores: {}".format(feature, np.round(train_score, 5), np.round(valid_score, 5)))
                train_scores.append(train_score)
                valid_scores.append(valid_score)
        else:
            train_scores, valid_scores = self.cross_validation(X=np.array(df[feature_set]), y=np.array(df[[target_name]]).squeeze(), metric=metric, algorithm_instance=algorithm_instance, cross_validation_instance=self.cross_validation_instance)
            print("Mean of Train Scores: {}; Mean of Valid Scores: {}".format(np.round(train_scores, 5), np.round(valid_scores, 5)))
        self.summary_frame = self.summary_features(train_scores=train_scores, valid_scores=valid_scores, features=feature_set)
    
    def change_features_scale(self, df, metric, algorithm_instance, features_to_scale, target_name, scale, verbose=False):
        df = self.check_df(df=df)
        features_names = df.columns.tolist()
        features_names.remove(target_name)
        features_to_scale = self.check_feature_set(feature_set=features_to_scale, features_names=features_names)
        scale = self.check_scale(scale=scale)
        train_scores, valid_scores, features_and_scales = [], [], []
        all_combinations = [list(zip(each_permutation, features_to_scale)) for each_permutation in itertools.permutations(scale, len(features_to_scale))]
        for element in all_combinations:
            dict_full = dict()
            df_copy = df.copy()
            for index, (skala, feature_name) in enumerate(element):
                df_copy[feature_name] = df_copy[feature_name]*skala
                dict_full[f"{feature_name}_scale"] = skala
            train_score, valid_score = self.cross_validation(X=np.array(df_copy[features_names]), y=np.array(df[[target_name]]).squeeze(), metric=metric, algorithm_instance=algorithm_instance, cross_validation_instance=self.cross_validation_instance)
            if(verbose == True):
                string = ""
                for key, value in dict_full.items():
                    string += f"{key}: scale={value}; "
                string += f"Mean of Train Scores: {np.round(train_score, 5)}; Mean of Valid Scores: {np.round(valid_score, 5)}"
                print(string)
            train_scores.append(train_score)
            valid_scores.append(valid_score)
            features_and_scales.append(dict_full)
        self.summary_frame = self.summary_scale(train_scores=train_scores, valid_scores=valid_scores, features_and_scales=features_and_scales)
    
    def check_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Wrong type of df. It should be dataframe')
        return df
    
    def check_feature_set(self, feature_set, features_names):
        if not isinstance(feature_set, str) and not isinstance(feature_set, list) and not isinstance(feature_set, np.ndarray):
            raise TypeError('Wrong type of feature_set. It should be string, list or numpy array.')
        if(isinstance(feature_set, str) == True):
            feature_set = np.array([feature_set])
        for feature in feature_set:
            if feature not in features_names:
                raise TypeError(f'{feature} set provided contains column names that does not exist in df.')
        return feature_set

    def check_scale(self, scale):
        if not isinstance(scale, int) and not isinstance(scale, float) and not isinstance(scale, list) and not isinstance(scale, np.ndarray):
            raise TypeError('Wrong type of scale. It should be int, float, list or numpy array.')
        if(isinstance(scale, int) == True or isinstance(scale, float) == True):
            scale = np.array([scale])
        return scale

    def cross_validation(self, X, y, metric, algorithm_instance, cross_validation_instance):
        metrics = {"accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
                    "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
                    "neg_mse": [lambda y, y_pred: -mean_squared_error(y, y_pred), "preds"],
                    "neg_rmse": [lambda y, y_pred: -mean_squared_error(y, y_pred)**0.5, "preds"],
                    "neg_mae": [lambda y, y_pred: -mean_absolute_error(y, y_pred), "preds"]}
        eval_metric = metrics[metric][0]
        metric_type = metrics[metric][1]
        algorithm = algorithm_instance
        cv = cross_validation_instance
        train_scores, valid_scores = [], []
        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            algorithm.fit(X_train, y_train)
            if(metric_type == "preds"):
                y_train_pred = algorithm.predict(X_train)
                y_valid_pred = algorithm.predict(X_valid)
            else:
                y_train_pred = algorithm.predict_proba(X_train)[:, 1]
                y_valid_pred = algorithm.predict_proba(X_valid)[:, 1]
            train_scores.append(eval_metric(y_train, y_train_pred))
            valid_scores.append(eval_metric(y_valid, y_valid_pred))
        return np.mean(train_scores), np.mean(valid_scores)

    def summary_features(self, train_scores, valid_scores, features):
        final_dict = {"Feature": features, "Mean of Train Scores": train_scores, "Mean of Valid scores": valid_scores}
        df = pd.DataFrame(final_dict)
        df = df.sort_values(by=['Mean of Valid scores', 'Mean of Train Scores'], ascending=[False, False])
        df.reset_index(drop=True, inplace=True)
        return df

    def summary_scale(self, train_scores, valid_scores, features_and_scales):
        df = pd.DataFrame(features_and_scales)
        df["Mean of Train Scores"] = train_scores
        df["Mean of Valid scores"] = valid_scores
        df = df.sort_values(by=['Mean of Valid scores', 'Mean of Train Scores'], ascending=[False, False])
        df.reset_index(drop=True, inplace=True)
        return df