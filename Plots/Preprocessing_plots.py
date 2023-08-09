from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Preprocessing_plots():
    def __init__(self):
        pass

    def hist_plot_with_kde(self, data, xaxis_title):
        data = np.array(data)
        if(data.ndim==2):
            data = data.squeeze()
        hist_data = [data]
        group_labels =[xaxis_title]
        fig = ff.create_distplot(hist_data, group_labels=group_labels, show_rug=False, curve_type='kde', bin_size=1)
        fig.update_layout(template="simple_white", width=800, height=600, showlegend=False, xaxis_title=xaxis_title, yaxis_title="Density", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def boxplot(self, data):
        data = np.array(data)
        if(data.ndim == 2):
            data = data.squeeze()
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, showlegend=False))
        fig.update_layout(template="simple_white", width=800, xaxis_title="", height=600, showlegend=False, yaxis_title="Sample data", font=dict(family="Times New Roman",size=16,color="Black"))
        fig.show("png")
    
    def correlation_plot(self, df, features_names):
        df = np.array(df)
        df = pd.DataFrame(df, columns=features_names)

        corr = np.round(df[df.columns.tolist()].corr(), 3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        df_mask = corr.mask(mask)
        fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), x=df_mask.columns.tolist(), y=df_mask.columns.tolist(),colorscale=px.colors.diverging.RdBu,hoverinfo="none", showscale=True, ygap=1, xgap=1)
        fig.update_xaxes(side="bottom")
        fig.update_layout(width=1200, height=800,xaxis_showgrid=False,yaxis_showgrid=False,xaxis_zeroline=False,yaxis_zeroline=False,yaxis_autorange='reversed',template='plotly_white',font=dict(family="Times New Roman",size=12,color="Black"))
        # NaN values are not handled automatically and are displayed in the figure
        # So we need to get rid of the text manually
        for i in range(len(fig.layout.annotations)):
            if fig.layout.annotations[i].text == 'nan':
                fig.layout.annotations[i].text = ""
        fig.show("png")