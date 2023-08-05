from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Regression_metrics import *
from Metrics.Classification_metrics import *
from Linear_Regression.Model.Linear_regression_from_scratch import *

class Algorithm_plots():
    def __init__(self):
        pass

    #Linear Regression Plots
    def plot_linear_regression(self, y_true, y_pred, metric="MSE"):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if(y_true.ndim == 2):
            y_true=y_true.squeeze()
        if(y_pred.ndim == 2):
            y_pred=y_pred.squeeze()
        self.metric = metric
        metrics = { "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": root_mean_squared_error(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "MAPE": mean_absolute_percentage_error(y_true, y_pred),
                    "MedAE": median_absolute_error(y_true, y_pred),
                    "MSLE": mean_squared_logarithm_error(y_true, y_pred)}
        if self.metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.eval_metric = np.round(metrics[self.metric], 5)
        model = Linear_Regression(fit_intercept=True, optimization=False, degree=2)
        model.fit(y_true, y_pred, features_names=["y_pred"], target_name="Sales", robust=False)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=y_true, mode='markers',marker=dict(colorscale='Jet', size=10, line=dict(color='black', width=1)), showlegend=False))
        fig.add_trace(go.Scatter(x=y_pred, y=model.coef_[1]*y_pred+model.coef_[0], mode='lines',line=dict(color='#e10c00', width=5), showlegend=False))
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, xaxis_title="Predictions", yaxis_title="Values", font=dict(family="Times New Roman",size=16,color="Black"), title_text="<b>Mean squared error: {}<b>".format(np.round(mean_squared_error(y_true, y_pred), 4)), title_x=0.5, title_y=0.97)
        fig.show("png")

    def homoscedacity_plot(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if(y_true.ndim == 2):
            y_true=y_true.squeeze()
        if(y_pred.ndim == 2):
            y_pred=y_pred.squeeze()
        residuals = y_true - y_pred
        fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Fitted values",  "y": "Residuals"}, trendline="lowess", trendline_color_override="green")
        fig.add_shape(type="line", line_color="red", line_width=2, opacity=0.5, line_dash="dash",x0=0, x1=1, xref="paper", y0=0, y1=0, yref="y")
        fig.update_layout(template="simple_white", width=800, height=600, showlegend=False, xaxis_title="Fitted values",yaxis_title="Residuals", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def influence_plot(self, X, outliers):
        fig = go.Figure()
        fig.add_shape(type="line", x0=np.min(outliers.leverage)-np.quantile(outliers.leverage, 0.01), y0=2, x1=np.max(outliers.leverage)+np.quantile(outliers.leverage, 0.01), y1=2,line=dict(color="red",width=4,dash="dash"))
        fig.add_shape(type="line", x0=np.min(outliers.leverage)-np.quantile(outliers.leverage, 0.01), y0=-2, x1=np.max(outliers.leverage)+np.quantile(outliers.leverage, 0.01), y1=-2,line=dict(color="red",width=4,dash="dash"))
        fig.add_shape(type="line", x0=2*X.shape[1]/X.shape[0], y0=np.min(outliers.standarized_residuals)+np.quantile(outliers.standarized_residuals, 0.25), x1=2*X.shape[1]/X.shape[0], y1=np.max(outliers.standarized_residuals)-np.quantile(outliers.standarized_residuals, 0.25),line=dict(color="green",width=4,dash="dash"))
        fig.add_trace(go.Scatter(x=outliers.leverage, y=outliers.standarized_residuals, mode="markers", marker=dict(size=outliers.cook_distance, sizemode="area", sizeref=2.*max(outliers.cook_distance)/(40.**2), sizemin=4)))
        i = 0
        while(i < len(outliers.indices_of_outliers)):
            fig.add_annotation(x=outliers.leverage[outliers.indices_of_outliers[i]], y=outliers.standarized_residuals[outliers.indices_of_outliers[i]],text=outliers.indices_of_outliers[i],showarrow=False)
            i = i + 1
        fig.update_layout(template="simple_white", width=800, height=600, showlegend=False, yaxis_range=[np.min(outliers.standarized_residuals)+np.quantile(outliers.standarized_residuals, 0.25), np.max(outliers.standarized_residuals)-np.quantile(outliers.standarized_residuals, 0.25)], xaxis_range=[np.min(outliers.leverage)-np.quantile(outliers.leverage, 0.01),np.max(outliers.leverage)+np.quantile(outliers.leverage, 0.01)], xaxis_title="Leverage", yaxis_title="Standarized Residuals", title_text="<b>Influence plot<b>", title_x=0.5, title_y=0.99, font=dict(family="Times New Roman",size=16,color="Black"), margin=dict(l=0, r=0.5, t=0.5, b=0.5))
        fig.show("png")
    
    def quantile_plot(self, data):
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        qqplot_data = qqplot(data, line='s').gca().lines
        plt.close()
        fig = go.Figure()
        fig.add_trace({'type': 'scatter','x': qqplot_data[0].get_xdata(),'y': qqplot_data[0].get_ydata(), 'mode': 'markers', 'marker': {'color': 'blue'}})
        fig.add_trace({'type': 'scatter','x': qqplot_data[1].get_xdata(),'y': qqplot_data[1].get_ydata(),'mode': 'lines','line': {'color': 'black'}})
        fig.update_layout(template="simple_white", width=800, height=600, showlegend=False, xaxis_title="Theoritical Quantities",yaxis_title="Sample Quantities", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    #KNN Plots
    def k_neighbors_tuning_plot(self, n_neighbors, train_scores, valid_scores, metric):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[k for k in n_neighbors], y=train_scores, mode='lines', line=dict(color="orange"), name="Train Scores"))
        fig.add_trace(go.Scatter(x=[k for k in n_neighbors], y=valid_scores, mode='lines', line=dict(color="blue"), name="Valid Scores"))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>K Neighbors Tuning<b>", title_x=0.5, xaxis_title="Number of neighbors", yaxis_title=f"{metric.upper()}", font=dict(family="Times New Roman",size=16,color="Black"), showlegend=True)
        fig.show("png")
    
    #DecisionTree Plots
    def depth_alpha_plot(self, ccp_alphas, depths):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ccp_alphas, y=depths, mode='lines+markers', line=dict(color="blue")))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Depth vs Alpha<b>", title_x=0.5, xaxis_title="Alpha", yaxis_title=f"Depth of tree", font=dict(family="Times New Roman",size=16,color="Black"), showlegend=False)
        fig.show("png")
    
    def nodes_alpha_plot(self, ccp_alphas, nodes):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ccp_alphas, y=nodes, mode='lines+markers', line=dict(color="blue")))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Nodes vs Alpha<b>", title_x=0.5, xaxis_title="Alpha", yaxis_title=f"Number of nodes", font=dict(family="Times New Roman",size=16,color="Black"), showlegend=False)
        fig.show("png")
    
    def scores_alpha_plot(self, ccp_alphas, train_scores, valid_scores, metric_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ccp_alphas, y=train_scores, mode='lines+markers', line=dict(color="orange"), name="Train Scores"))
        fig.add_trace(go.Scatter(x=ccp_alphas, y=valid_scores, mode='lines+markers', line=dict(color="blue"), name="Valid Scores"))
        index_of_best_valid_score = valid_scores.index(max(valid_scores))
        fig.add_annotation(x=ccp_alphas[index_of_best_valid_score], y=valid_scores[index_of_best_valid_score], showarrow=True, arrowhead=1, text=f"ccp_alpha={np.round(ccp_alphas[index_of_best_valid_score], 5)}", font=dict(family="Times New Roman",size=12,color="Black"))
        fig.update_layout(template="simple_white", width=600, height=600, title=f"<b>{metric_name} vs alpha<b>", title_x=0.5, xaxis_title="Alpha", yaxis_title=f"{metric_name}", font=dict(family="Times New Roman",size=16,color="Black"), showlegend=True)
        fig.show("png")