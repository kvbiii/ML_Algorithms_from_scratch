from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Classification_metrics import *
from Metrics.Regression_metrics import *

class Prediction_plots():
    def __init__(self):
        pass
    def compare_predictions_with_real_values(self, y_true, y_pred, metric="MSE"):
        self.metric = metric
        y_true = np.array(y_true)
        if(y_true.ndim == 2):
            y_true = y_true.squeeze()
        y_pred = np.array(y_pred)
        if(y_pred.ndim == 2):
            y_pred = y_pred.squeeze()
        metrics = { "MSE": mean_squared_error(y_true, y_pred),
                    "RMSE": root_mean_squared_error(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "MAPE": mean_absolute_percentage_error(y_true, y_pred),
                    "MedAE": median_absolute_error(y_true, y_pred),
                    "MSLE": mean_squared_logarithm_error(y_true, y_pred)}
        if self.metric not in metrics:
            raise ValueError('Unsupported metric: {}'.format(metric))
        self.eval_metric = np.round(metrics[self.metric], 5)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(len(y_true))], y=y_true.flatten().tolist(), mode='lines', line=dict(color="orange"), name="Real values"))
        fig.add_trace(go.Scatter(x=[i for i in range(len(y_true))], y=y_pred.flatten().tolist(), mode='lines', line=dict(color="blue"), name="Predictions"))
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Predictions and Real values<b>", title_x=0.5, xaxis_title="Observation", yaxis_title="Values", font=dict(family="Times New Roman",size=16,color="Black"), legend_title_text='{}: {}'.format(self.metric.upper(), self.eval_metric))
        fig.show("png")

    def plot_losses(self, train_loss, valid_loss=None):
        if(train_loss.ndim == 2):
            train_loss = train_loss.squeeze()
        if(valid_loss != None):
            if(valid_loss.ndim == 2):
                valid_loss = valid_loss.squeeze()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(len(train_loss))], y=train_loss, mode='lines', line=dict(color="orange"), name="Train loss"))
        try:
            fig.add_trace(go.Scatter(x=[i for i in range(len(valid_loss))], y=valid_loss, mode='lines', line=dict(color="blue"), name="Valid loss"))
        except:
            pass
        fig.update_layout(template="simple_white", width=600, height=600, title="<b>Loss plot<b>", title_x=0.5, xaxis_title="Epochs", yaxis_title="Loss", font=dict(family="Times New Roman",size=16,color="Black"), legend=dict(yanchor="top",y=0.85,xanchor="left",x=0.65))
        fig.show("png")

    def conf_matrix(self, y_true, y_pred, normalize=False):
        if(normalize==True):
            CM = confusion_matrix(y_true, y_pred, normalize='true')
        else:
            CM = confusion_matrix(y_true, y_pred, normalize=None)
        TN = np.round(CM[0][0], 3)
        FN = np.round(CM[1][0], 3)
        TP = np.round(CM[1][1], 3)
        FP = np.round(CM[0][1], 3)
        z = [[TN, FP],
            [FN, TP]]
        z_text = [[str(y) for y in x] for x in z]
        x = ['0', '1']
        y =  ['0', '1']
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='blues')
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, font=dict(family="Times New Roman",size=16,color="Black"), title_text="<b>Normalized confusion matrix<br>Balanced accuracy score: {}<b>".format(np.round(balanced_accuracy_score(y_true, y_pred), 4)), title_x=0.5, title_y=0.97)
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=-0.15,y=0.5,showarrow=False,text="Actual",textangle=-90,xref="paper",yref="paper"))
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=0.5,y=1.1,showarrow=False,text="Predictions",xref="paper",yref="paper"))
        fig['data'][0]['showscale'] = True
        fig.show("png")

    def multilabel_conf_matrix(self, y_true, y_pred, labels, normalize=False):
        if(normalize==True):
            CM = confusion_matrix(y_true, y_pred, normalize='true')
        else:
            CM = confusion_matrix(y_true, y_pred, normalize=None)
        z = []
        for i in range(len(labels)):
            z.append([])
            for j in range(len(labels)):
                z[i].append(np.round(CM[i][j], 3))
        z_text = [[str(y) for y in x] for x in z]
        x = [i for i in labels]
        y =  [i for i in labels]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='blues')
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, font=dict(family="Times New Roman",size=16,color="Black"), title_text="<b>Normalized confusion matrix<br>Balanced accuracy score: {}<b>".format(np.round(balanced_accuracy_score(y_true, y_pred), 4)), title_x=0.5, title_y=0.97)
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=-0.15,y=0.5,showarrow=False,text="Actual",textangle=-90,xref="paper",yref="paper"))
        fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=0.5,y=1.1,showarrow=False,text="Predictions",xref="paper",yref="paper"))
        fig['data'][0]['showscale'] = True
        fig.show("png")

    def roc_auc_plot(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color="orange"), name="ROC Curve: {}".format(np.round(roc_auc_score(y_true, y_prob), 4))))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color="blue", dash='dash'), showlegend=False))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", font=dict(family="Times New Roman",size=16,color="Black"), legend=dict(yanchor="top",y=0.25,xanchor="left",x=0.65))
        fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1, linecolor='black')
        fig.update_xaxes(range=[0, 1], constrain='domain', linecolor='black')
        fig.show("png")
    
    def precision_recall_plot(self, y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color="blue"), fill='tonexty', fillcolor='cornflowerblue', showlegend=True, name="PR Curve: {}".format(np.round(average_precision_score(y_true, y_prob), 4))))
        fig.update_layout(template="simple_white", width=600, height=600, xaxis_title="Recall",yaxis_title="Precision", font=dict(family="Times New Roman",size=16,color="Black"), legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.65))
        fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1, linecolor='black')
        fig.update_xaxes(range=[0, 1], constrain='domain', linecolor='black')
        fig.show("png")

    def gain_plot(self, y_true, y_prob):
        data = kds.metrics.decile_table(y_true, y_prob)
        decile = [0]+data["decile"].values.flatten().tolist()
        gain_pct = [0]+data["cum_resp_pct"].values.flatten().tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decile, y=gain_pct, mode='lines+markers', line=dict(color="purple"), name="Gain"))
        fig.add_trace(go.Scatter(x=[0, 10], y=[0, 100], mode='lines', line=dict(color="blue", dash='dash'), name="Random"))
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=True, xaxis_title="Decile",yaxis_title="Number of respondents (%)", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.update_yaxes(range=[0, 100], linecolor='black', dtick=20)
        fig.update_xaxes(range=[0, 10], constrain='domain', linecolor='black', dtick=1)
        fig.show("png")
    
    def lift_plot(self, y_true, y_prob):
        data = kds.metrics.decile_table(y_true, y_prob)
        decile = data["decile"].values.flatten().tolist()
        lift_pct = data["lift"].values.flatten().tolist()
        random = [1 for i in range(0, len(decile))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decile, y=lift_pct, mode='lines+markers', line=dict(color="green"), name="Model"))
        fig.add_trace(go.Scatter(x=decile, y=random, mode='lines+markers', line=dict(color="blue"), name="Random"))
        fig.update_layout(template="simple_white", width=600, height=600, showlegend=True, xaxis_title="Decyl",yaxis_title="Lift", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.update_xaxes(linecolor='black', dtick=1)
        fig.show()

    def plot_feature_importances(self, feature_importances, column_names=None, nlargest=None):
        if(column_names == None):
            column_names = np.array([i for i in range(len(feature_importances))])
        if not isinstance(column_names, np.ndarray):
            try:
                column_names = np.array(column_names)
            except:
                raise TypeError('Wrong type of column_names. It should be numpy array, or list..')
        if(nlargest != None):
            ranking = (np.argsort(np.argsort(-np.array(feature_importances))))
            support = np.where(ranking < nlargest, True, False)
            feature_importances = feature_importances[support]
            column_names = column_names[support]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=column_names, y=feature_importances, marker_color='rgb(26, 118, 255)'))
        fig.update_layout(template="simple_white", width=max(30*len(column_names), 600), height=max(30*len(column_names), 600), title_text="<b>Feature importance<b>", title_x=0.5, yaxis_title="Feature importance", xaxis=dict(title='Features', showticklabels=True, type="category"), font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")