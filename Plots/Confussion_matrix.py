from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
pio.renderers.default = "plotly_mimetype+notebook"
def conf_matrix(y_true, predictions, normalize=False):
    if(normalize==True):
        CM = confusion_matrix(y_true, predictions, normalize='true')
    else:
        CM = confusion_matrix(y_true, predictions, normalize=None)
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
    fig.update_layout(template="simple_white", width=600, height=600, showlegend=False, font=dict(family="Times New Roman",size=16,color="Black"), title_text="<b>Normalized confusion matrix<br>Balanced accuracy score: {}<b>".format(np.round(balanced_accuracy_score(y_true, predictions), 4)), title_x=0.5, title_y=0.97)
    fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=-0.15,y=0.5,showarrow=False,text="Actual",textangle=-90,xref="paper",yref="paper"))
    fig.add_annotation(dict(font=dict(family="Times New Roman",size=20,color="Black"),x=0.5,y=1.1,showarrow=False,text="Predictions",xref="paper",yref="paper"))
    fig['data'][0]['showscale'] = True
    fig.show("png")