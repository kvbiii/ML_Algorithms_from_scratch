from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

class Create_Interactions():
    def __init__(self):
        pass
    
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
            j = i+1
            while(j < self.X_original_len):
                self.X = np.column_stack([self.X, self.X[:,i]*self.X[:,j]])
                j = j + 1
            i = i + 1
        return self.X
    
    def get_feature_names_out(self, input_features_names):
        self.features_names = input_features_names
        i = 0
        while(i < self.X_original_len):
            j = i+1
            while(j < self.X_original_len):
                self.features_names.append(f"{self.features_names[i]}*{self.features_names[j]}")
                j = j + 1
            i = i + 1
        return self.features_names
    
    def drop_features_that_are_interaction_of_the_same_feature(self, X, features_names):
        indices_of_single_features = np.intersect1d(np.where(np.char.find(features_names, "*")==-1), np.where(np.char.find(features_names, "^")==-1))
        indices_to_drop = np.array([])
        i = 0
        while(i < len(indices_of_single_features)):
            indices_to_drop = np.append(indices_to_drop, np.where((np.char.count(features_names, f"{features_names[i]}")>=2))[0]).astype(int)
            i = i + 1
        X = np.delete(X, indices_to_drop, axis=1)
        features_names = np.delete(features_names, indices_to_drop, axis=0)
        indices_of_features_with_interactions_degrees = np.intersect1d(np.where(np.char.find(features_names, "*")!=-1), np.where(np.char.find(features_names, "^")!=-1)).astype(int)
        X = np.delete(X, indices_of_features_with_interactions_degrees, axis=1)
        features_names = np.delete(features_names, indices_of_features_with_interactions_degrees, axis=0)
        indices_to_drop = np.array([])
        i = 0
        while(i < len(features_names)):
            if(len(np.unique(X[:,i])) <= 2 and "^" in features_names[i]):
                indices_to_drop = np.append(indices_to_drop, i).astype(int)
            i = i + 1
        if(len(indices_to_drop) > 0):
            X = np.delete(X, indices_to_drop, axis=1)
            features_names = np.delete(features_names, indices_to_drop, axis=0)
        return X, features_names.flatten().tolist()