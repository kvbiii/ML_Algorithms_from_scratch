from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class PolynomialFeatures():
    def __init__(self, degree=3):
        self.degree = degree

    def fit_transform(self, X):
        self.X = X
        if not isinstance(self.X, np.ndarray):
            try:
                self.X = np.array(self.X)
            except:
                raise TypeError('Wrong type of X. It should be numpy array, torch_tensor, list or dataframe columns.')
        self.X_original_len = X.shape[1]
        i = 0
        while(i < self.X_original_len):
            j = 2
            while(j <= self.degree):
                new_variable = self.X[:,i]**j
                self.X = np.column_stack([self.X, new_variable])
                j = j + 1
            i = i + 1
        return self.X

    def get_feature_names_out(self, input_features_names):
        self.features_names = input_features_names
        i = 0
        while(i < self.X_original_len):
            j = 2
            while(j <= self.degree):
                self.features_names.append(f"{input_features_names[i]}^{j}")
                j = j + 1
            i = i + 1
        return self.features_names