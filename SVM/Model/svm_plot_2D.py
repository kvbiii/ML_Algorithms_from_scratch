from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Model.svm_smo import *
class svm_plots_with_test():
    def __init__(self, X_train, y_train, X_test, y_test, values, param_name, kernel):
        self.X_train = X_train
        self.y_train = y_train
        self.values = values
        self.param_name = param_name
        self.kernel = kernel
        self.X_test = X_test
        self.y_test = y_test
    
    def model_fitting(self, params):
        model = Support_Vector_Machines_SMO(**params, kernel=self.kernel)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return model, np.round(balanced_accuracy_score(self.y_test, y_pred), 4)
    
    def plotting_each_trace_train(self, lims, Z, legenda, model):
        support_vectors = go.Scatter(x=model.support_vectors_[:, 0], y=model.support_vectors_[:, 1], mode='markers', marker=dict(color='white', size=20, line=dict(color='black', width=1)), showlegend=False)
        scatter = go.Scatter(x=self.X_train[:,0], y=self.X_train[:,1], mode='markers',marker=dict(color=self.y_train, colorscale='Jet', size=10, line=dict(color='black', width=1)), showlegend=False)
        hyperplane = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=0), name='Hyperplane',line=dict(color='black'), showlegend=legenda)
        support_line_1 = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=1), name='Support line for class 1',line=dict(color='red', dash='dash'), showlegend=legenda)
        support_line_2 = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=-1), name='Support line for class 0',line=dict(color='blue', dash='dash'), showlegend=legenda)
        fig = go.Figure(data=[support_vectors, scatter, hyperplane, support_line_1, support_line_2])
        return fig
    
    def plotting_each_trace_test(self, lims, Z):
        scatter = go.Scatter(x=self.X_test[:,0], y=self.X_test[:,1], mode='markers',marker=dict(color=self.y_test, colorscale='Jet', size=10, line=dict(color='black', width=1)), showlegend=False)
        hyperplane = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=0), name='Hyperplane',line=dict(color='black'), showlegend=False)
        support_line_1 = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=1), name='Support line for class 1',line=dict(color='red', dash='dash'), showlegend=False)
        support_line_2 = go.Contour(x=lims, y=lims, z=Z, contours=dict(showlines=True, type='constraint', operation='=',value=-1), name='Support line for class 0',line=dict(color='blue', dash='dash'), showlegend=False)
        fig = go.Figure(data=[scatter, hyperplane, support_line_1, support_line_2])
        return fig

    def create_subplot(self):
        rows = len(self.values)//2+1
        if(2*len(self.values) < 4):
            columns = 2*len(self.values)
        else:
            columns = 4
        lista = []
        i = 0
        while(i < len(self.values)):
            param = {self.param_name: self.values[i]}
            _, score = self.model_fitting(param)
            lista.append("")
            lista.append("{}={}<br>BACC: {}".format(self.param_name, self.values[i], score))
            i = i + 1
        tup = tuple(lista)
        final_fig = make_subplots(rows=rows, cols=columns, subplot_titles=tup)
        i = 0
        while(i < len(self.values)):
            param = {self.param_name: self.values[i]}
            model, _ = self.model_fitting(param)
            lims = np.linspace(start=min(min(self.X_train[:, 0]), min(self.X_train[:, 1]))-1, stop=max(max(self.X_train[:, 0]), max(self.X_train[:, 1]))+1, num=500)
            x1, x2 = np.meshgrid(lims, lims)
            Z = model.decision_function(np.c_[x1.ravel(), x2.ravel()])
            Z = Z.reshape(x1.shape)
            if(i==0):
                legenda = True
            else:
                legenda = False
            fig = self.plotting_each_trace_train(lims=lims, Z=Z, legenda=legenda, model=model)
            row = i//2+1
            col = (i%2+1)*2-1
            for trace in fig.data:
                final_fig.append_trace(trace, row=row, col=col)
            col = col + 1
            fig = self.plotting_each_trace_test(lims=lims, Z=Z)
            for trace in fig.data:
                final_fig.append_trace(trace, row=row, col=col)
            i = i + 1
        width = columns*1000
        height = rows*1000
        final_fig.update_layout(template="simple_white", width=width, height=height, showlegend=True, font=dict(family="Times New Roman",size=16,color="Black"))
        final_fig.show(renderer="browser")