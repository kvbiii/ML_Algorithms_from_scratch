from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from Plots.Algorithms_plots import *
Algorithms_plots = Algorithm_plots()

class Cross_Validation_Decision_Tree_post_pruning():
    def __init__(self, metric, algorithm_instance, cross_validation_instance):
        metrics = {"accuracy": [lambda y, y_pred: accuracy_score(y, y_pred), "preds"],
                    "roc_auc": [lambda y, y_pred: roc_auc_score(y, y_pred), "probs"],
                    "mse": [lambda y, y_pred: mean_squared_error(y, y_pred), "preds"],
                    "rmse": [lambda y, y_pred: mean_squared_error(y, y_pred)**0.5, "preds"],
                    "mae": [lambda y, y_pred: mean_absolute_error(y, y_pred), "preds"]}
        if metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.metric_name = metric
        self.eval_metric = metrics[metric][0]
        self.metric_type = metrics[metric][1]
        self.algorithm = algorithm_instance
        self.cv = cross_validation_instance
    
    def find_alphas(self, X, y):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        path = self.algorithm.cost_complexity_pruning_path(X, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        return ccp_alphas[:-5], impurities[:-5]
    
    def compare_depth_and_alpha(self, X, y):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        ccp_alphas, impurities = self.find_alphas(X=X, y=y)
        ccp_alphas_limited = [np.quantile(ccp_alphas, q) for q in np.linspace(0, 1, 15)]
        depths = []
        for ccp_alpha in ccp_alphas_limited:
            algorithm = self.algorithm.set_params(ccp_alpha=ccp_alpha)
            depths.append(algorithm.fit(X, y).tree_.max_depth)
        Algorithms_plots.depth_alpha_plot(ccp_alphas=ccp_alphas_limited, depths=depths)
    
    def compare_nodes_and_alpha(self, X, y):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        ccp_alphas, impurities = self.find_alphas(X=X, y=y)
        ccp_alphas_limited = [np.quantile(ccp_alphas, q) for q in np.linspace(0, 1, 15)]
        nodes = []
        for ccp_alpha in ccp_alphas_limited:
            algorithm = self.algorithm.set_params(ccp_alpha=ccp_alpha)
            nodes.append(algorithm.fit(X, y).tree_.node_count)
        Algorithms_plots.nodes_alpha_plot(ccp_alphas=ccp_alphas_limited, nodes=nodes)
    
    def compare_scores_and_alpha(self, X, y, verbose=False):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        ccp_alphas, impurities = self.find_alphas(X=X, y=y)
        ccp_alphas_limited = [np.quantile(ccp_alphas, q) for q in np.linspace(0, 1, 15)]
        all_mean_train_scores, all_mean_valid_scores = [], []
        for ccp_alpha in ccp_alphas_limited:
            train_scores, valid_scores = [], []
            algorithm = self.algorithm.set_params(ccp_alpha=ccp_alpha)
            for iter, (train_idx, valid_idx) in enumerate(self.cv.split(X, y)):
                X_train, X_valid = X[train_idx, :], X[valid_idx, :]
                y_train, y_valid = y[train_idx], y[valid_idx]
                algorithm.fit(X_train, y_train)
                if(self.metric_type == "preds"):
                    y_train_pred = algorithm.predict(X_train)
                    y_valid_pred = algorithm.predict(X_valid)
                else:
                    y_train_pred = algorithm.predict_proba(X_train)[:, 1]
                    y_valid_pred = algorithm.predict_proba(X_valid)[:, 1]
                train_scores.append(self.eval_metric(y_train, y_train_pred))
                valid_scores.append(self.eval_metric(y_valid, y_valid_pred))
            if(verbose==True):
                print("ccp_alpha={}: train scores: {}; valid scores: {}".format(np.round(ccp_alpha, 5), np.round(np.mean(train_scores), 5), np.round(np.mean(valid_scores), 5)))
            all_mean_train_scores.append(np.mean(train_scores))
            all_mean_valid_scores.append(np.mean(valid_scores))
        Algorithms_plots.scores_alpha_plot(ccp_alphas=ccp_alphas_limited, train_scores=all_mean_train_scores, valid_scores=all_mean_valid_scores, metric_name=self.metric_name)
    
    def cv_for_range_of_alpha(self, X, y, min_alpha, max_alpha, verbose=False):
        X = self.check_X(X=X)
        y = self.check_y(y=y)
        ccp_alphas, impurities = self.find_alphas(X=X, y=y)
        ccp_alphas_limited = [alpha for alpha in ccp_alphas if alpha >= min_alpha and alpha <= max_alpha]
        all_mean_train_scores, all_mean_valid_scores = [], []
        for ccp_alpha in ccp_alphas_limited:
            train_scores, valid_scores = [], []
            algorithm = self.algorithm.set_params(ccp_alpha=ccp_alpha)
            for iter, (train_idx, valid_idx) in enumerate(self.cv.split(X, y)):
                X_train, X_valid = X[train_idx, :], X[valid_idx, :]
                y_train, y_valid = y[train_idx], y[valid_idx]
                algorithm.fit(X_train, y_train)
                if(self.metric_type == "preds"):
                    y_train_pred = algorithm.predict(X_train)
                    y_valid_pred = algorithm.predict(X_valid)
                else:
                    y_train_pred = algorithm.predict_proba(X_train)[:, 1]
                    y_valid_pred = algorithm.predict_proba(X_valid)[:, 1]
                train_scores.append(self.eval_metric(y_train, y_train_pred))
                valid_scores.append(self.eval_metric(y_valid, y_valid_pred))
            if(verbose==True):
                print("ccp_alpha={}: train scores: {}; valid scores: {}".format(np.round(ccp_alpha, 5), np.round(np.mean(train_scores), 5), np.round(np.mean(valid_scores), 5)))
            all_mean_train_scores.append(np.mean(train_scores))
            all_mean_valid_scores.append(np.mean(valid_scores))
        Algorithms_plots.scores_alpha_plot(ccp_alphas=ccp_alphas_limited, train_scores=all_mean_train_scores, valid_scores=all_mean_valid_scores, metric_name=self.metric_name)
    
    def check_X(self, X):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be dataframe, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        return y