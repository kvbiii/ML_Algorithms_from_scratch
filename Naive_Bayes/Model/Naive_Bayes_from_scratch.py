from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class Naive_Bayes():
    def __init__(self, alpha=1):
        #self.categorical_features = categorical_features
        self.alpha = alpha
        pass

    def fit(self, X, y):
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.unique_classes = np.unique(y)
        self.categorical_variables = self.decide_type_of_variables(X=self.X_train)
        self.class_count_dictionary = self.calculate_count_for_each_class(y=self.y_train)
        self.class_prior_dictionary = self.calculate_prior_probabilities_for_each_class(class_count_dict=self.class_count_dictionary)
        self.category_evidence = self.calculate_evidence_for_each_category(X=self.X_train, categorical_variables=self.categorical_variables)
        self.likelihood = self.calculate_likelihood_for_given_vector(X=self.X_train, y=self.y_train, categorical_variables=self.categorical_variables)
        self.mean_variable, self.std_variable, self.mean_variable_class, self.std_variable_class = self.calculate_mean_and_std_for_continous_variables(X=self.X_train, y=self.y_train, categorical_variables=self.categorical_variables)
        self.fit_used = True
    
    def check_X(self, X, train):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        X = np.array(X)
        if(X.ndim == 1):
            X = X[None, :]
        if(train == False):
            if(self.X_train.shape[1] != X.shape[1]):
                raise ValueError(f"X has {X.shape[1]} features, but Naive Bayes is expecting {self.X_train.shape[1]} features as input.")
        return X
    
    def check_y(self, y):
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y)
        if(y.ndim == 2):
            y = y.squeeze()
        if(len(np.unique(y)) > 30 or (self.X_train.shape[0] < 30 and len(np.unique(y)) == self.X_train.shape[0])):
            raise TypeError('Naive Bayes is for classification tasks. Target data provided is continous variable while it should be categorical or binary.')
        return y
    
    def decide_type_of_variables(self, X):
        categorical_variables = []
        for variable in range(0, X.shape[1]):
            if(len(np.unique(X[:,variable])) < 15):
                categorical_variables.append(variable)
        return categorical_variables

    def calculate_count_for_each_class(self, y):
        return {klasa: count_of_class for klasa, count_of_class in zip(np.unique(y, return_counts=True)[0], np.unique(y, return_counts=True)[1])}
    
    def calculate_prior_probabilities_for_each_class(self, class_count_dict):
        return {klasa: count_of_class/sum(class_count_dict.values()) for klasa, count_of_class in class_count_dict.items()}
    
    def calculate_evidence_for_each_category(self, X, categorical_variables):
        category_evidence_list = []
        for variable in range(0, X.shape[1]):
            if(variable in categorical_variables):
                category_evidence_list.append({category: count_of_category/X.shape[0] for category, count_of_category in zip(np.unique(X[:,variable], return_counts=True)[0], np.unique(X[:,variable], return_counts=True)[1])})
            else:
                category_evidence_list.append("Continous variable")
        return category_evidence_list
    
    def calculate_likelihood_for_given_vector(self, X, y, categorical_variables):
        #It looks like this: [{class_0: {category_0: (len(X_0==category_0 and y==class_0)+alpha)/(len(y=class_0)+alpha*X.shape[1]), category_1: (len(X_0==category_1 and y==class_0)+alpha)/(len(y=class_0)+alpha*X.shape[1])}, class_1: {...}}, {class_0: {...}},...]
        occurrences_of_given_category_for_variables_within_given_class = []
        for variable in range(0, X.shape[1]):
            if(variable in categorical_variables):
                occurrences_of_given_category_for_variables_within_given_class.append({klasa: {category: ((np.where((X[:,variable]==category) & (y==klasa))[0]).shape[0]+self.alpha)/(((np.where(y==klasa))[0]).shape[0]+self.alpha*X.shape[1]) for category in np.unique(X[:,variable], return_counts=True)[0]} for klasa in np.unique(y)})
            else:
                occurrences_of_given_category_for_variables_within_given_class.append("Continous variable")
        return occurrences_of_given_category_for_variables_within_given_class

    def calculate_mean_and_std_for_continous_variables(self, X, y, categorical_variables):
        mean_variable, std_variable, mean_variable_class, std_variable_class = [], [], [], []
        for variable in range(0, X.shape[1]):
            if(variable in categorical_variables):
                mean_variable.append("Categorical variable")
                std_variable.append("Categorical variable")
                mean_variable_class.append("Categorical variable")
                std_variable_class.append("Categorical variable")
            else:
                mean_variable.append(np.mean(X[:,variable]))
                std_variable.append(np.std(X[:,variable]))
                mean_variable_class.append({klasa: np.mean(X[np.where(y==klasa)[0], variable]) for klasa in np.unique(y)})
                std_variable_class.append({klasa: np.std(X[np.where(y==klasa)[0], variable]) for klasa in np.unique(y)})
        return mean_variable, std_variable, mean_variable_class, std_variable_class

    def predict(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        probabilities = self.get_proba(X=X)
        dict_for_mapping = {key: value for key, value in enumerate(self.unique_classes)}
        predicted_indices_of_classes = np.argmax(probabilities, axis=1)
        final_predictions = np.array([dict_for_mapping[elemenet] for elemenet in predicted_indices_of_classes])
        return final_predictions
    
    def predict_proba(self, X):
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        probabilities = self.get_proba(X=X)
        return probabilities

    def check_fit(self, fit_used):
        if fit_used == False:
            raise AttributeError('Naive Bayes has to be fitted first.')
    
    def get_proba(self, X):
        probabilities = np.zeros(shape=(X.shape[0], len(self.unique_classes)))
        product_of_likelihood = 0
        for row_iter in range(0, X.shape[0]):
            for class_iter, target in enumerate(self.unique_classes):
                prior = self.class_prior_dictionary[target]
                product_of_likelihood = 1
                prudct_of_evidence = 1
                for variable in range(0, X.shape[1]):
                    if(variable in self.categorical_variables):
                        product_of_likelihood *= self.likelihood[variable][target][X[row_iter,variable]]
                        prudct_of_evidence *= self.category_evidence[variable][X[row_iter,variable]]
                    else:
                        product_of_likelihood *= 1/np.sqrt(2*np.pi*self.std_variable_class[variable][target]**2)*np.exp(-1/2*((X[row_iter, variable]-self.mean_variable_class[variable][target])/self.std_variable_class[variable][target])**2)
                        prudct_of_evidence *= 1/np.sqrt(2*np.pi*self.std_variable[variable]**2)*np.exp(-1/2*((X[row_iter, variable]-self.mean_variable[variable])/self.std_variable[variable])**2)
                probabilities[row_iter, class_iter] = prior*product_of_likelihood/prudct_of_evidence
        scaled_probabilities = np.array([prob/np.sum(probabilities, axis=1)[line] for prob, line in zip(probabilities, range(0, probabilities.shape[0]))])
        return scaled_probabilities