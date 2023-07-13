from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class KNeighborsOptimization():
    def __init__(self, cross_validation_instance=KFold(n_splits=5, shuffle=True, random_state=17)):
        self.cross_validation_instance = cross_validation_instance

    def optimize(self, X, y, metric, n_neighbors=5, distance="euclidean", verbose=False):
        X = self.check_X(X=X)
        y = self.check_y(y=y, X=X)
        self.metric = metric
        self.n_neighbors = self.check_neighbors(n_neighbors=n_neighbors)
        distance = self.check_distance(distance=distance)
        algorithm_instance = self.algorithm_decider(problem_type=self.problem_type)
        self.train_scores, self.valid_scores, self.kwargs_permutation = [], [], []
        for n_neighbor, dist in itertools.product(self.n_neighbors, distance):
            kwargs = {"n_neighbors": n_neighbor, "metric": str(dist)}
            algorithm_instance = algorithm_instance.set_params(**kwargs)
            train_score, valid_score = self.cross_validation(X=X, y=y, metric=self.metric, algorithm_instance=algorithm_instance, cross_validation_instance=self.cross_validation_instance)
            if(verbose==True):
                print("Number of neighbors: {}; Distance: {}; Mean of Train Scores: {}; Mean of Valid Scores: {}".format(n_neighbor, dist, np.round(train_score, 5), np.round(valid_score, 5)))
            self.train_scores.append(train_score)
            self.valid_scores.append(valid_score)
            self.kwargs_permutation.append(kwargs)
        self.summary_frame = self.summary(train_scores=self.train_scores, valid_scores=self.valid_scores, kwargs=self.kwargs_permutation)
    
    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be dataframe, numpy array or torch tensor.')
        return np.array(X)
    
    def check_y(self, y, X):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        if(len(np.unique(y)) > 30 or (X.shape[0] < 30 and len(np.unique(y)) == X.shape[0])):
            self.problem_type = "regression"
        else:
            self.problem_type = "classification"
        return np.array(y)
    
    def check_neighbors(self, n_neighbors):
        if not isinstance(n_neighbors, int) and not isinstance(n_neighbors, list) and not isinstance(n_neighbors, np.ndarray) and not torch.is_tensor(n_neighbors):
            raise TypeError('Wrong type of n_neighbors. It should be int, list, numpy array or torch tensor.')
        if(isinstance(n_neighbors, int) == True):
            n_neighbors = np.array([n_neighbors])
        return n_neighbors

    def check_distance(self, distance):
        if not isinstance(distance, str) and not isinstance(distance, list) and not isinstance(distance, np.ndarray) and not torch.is_tensor(distance):
            raise TypeError('Wrong type of distance. It should be string, list, numpy array or torch tensor.')
        if(isinstance(distance, str) == True):
            distance = np.array([distance])
        return distance
    
    def algorithm_decider(self, problem_type):
        if(problem_type=="classification"):
            return KNeighborsClassifier()
        else:
            return KNeighborsRegressor()
    
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
    
    def summary(self, train_scores, valid_scores, kwargs):
        final_dict = {"Number of neighbors": list(map(lambda d: d['n_neighbors'], kwargs)), "Distance": list(map(lambda d: d['metric'], kwargs)), "Mean of Train Scores": train_scores, "Mean of Valid scores": valid_scores}
        df = pd.DataFrame(final_dict)
        df = df.sort_values(by=['Mean of Valid scores', 'Mean of Train Scores'], ascending=[False, False])
        df.reset_index(drop=True, inplace=True)
        return df
