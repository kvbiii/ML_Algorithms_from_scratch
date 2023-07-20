from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *
from Metrics.Classification_metrics import *

class Logistic_Regression():
    def __init__(self, fit_intercept=True, random_state=17, learning_rate=0.1, max_iter=500):
        """Initialize Logistic Regression class.

        Args:
            fit_intercept ([bool], optional): Specifies if a constant should be added to model. Defaults to True.
            random_state ([int], optional): Specifies seed. Defaults to 17.
            learning_rate ([float], optional): Specifies learning rate for gradient descent optimization. Defaults to 0.1.
            max_iter ([int], optional): Maximum number of iterations taken for the solvers to converge. Defaults to 500.
            fit_used ([bool]): Indicates whether our class has already been trained or not yet.
        """
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_used = False
    
    def fit(self, X, y, features_names=None, target_name=None):
        """Fit the model according to the given data.

        Args:
            X ([array-like]): Training vector.
            y ([array-like]): Target vector.
        """
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y)
        self.number_of_classes = self.check_number_of_classes(y=self.y_train)
        self.y_train = self.change_to_one_hot_encode(y=self.y_train)
        self.features_names, self.target_name = self.check_features_and_target_names(features_names=features_names, target_name=target_name)
        self.X_train = self.check_for_object_columns(X=self.X_train)
        self.X_train, self.features_names = self.check_intercept(X=self.X_train, features_names=self.features_names)
        self.coef_ = self.initialize_weights()
        self.find_coefficients_gradient_descent(X=self.X_train, y=self.y_train, learning_rate=self.learning_rate, max_iter=self.max_iter)
        self.fit_used = True

    def check_X(self, X, train):
        """Check type of input data and raise errors is something is wrong with it.
        
        Args:
            X ([array-like], shape=(n_samples, n_features)): input data.
            train ([bool]): True if this is array given in fit() method and False if this was argument for predict.

        Returns:
            np.array(X) ([np.array], shape=(n_samples, n_features)): numpy array of provided input data.
        """
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        if(train == False):
            if((self.fit_intercept == False and self.X_train.shape[1] != X.shape[1]) or (self.fit_intercept == True and self.X_train.shape[1] != X.shape[1]+1)):
                raise ValueError(f"X has {X.shape[1]} features, but Logistic Regression is expecting {self.X_train.shape[1]} features as input.")
        return np.array(X)
    
    def check_y(self, y):
        """Check type of target vector and raise errors is something is wrong with it.

        Args:
            y ([array-like], shape=(n_samples, )): target data.
            train ([bool]): True if this is array given in fit() method and False if this was argument for predict.

        Returns:
            np.array(y) ([np.array], shape=(n_samples, )) Numpy array of provided target variable.
        """
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        y = np.array(y, dtype=int)
        if(y.ndim == 2):
            y = y.squeeze()
        if(len(np.unique(y)) > 30 or (self.X_train.shape[0] < 30 and len(np.unique(y)) == self.X_train.shape[0])):
            raise TypeError('Logistic Regression is for classification tasks. Target data provided is continous variable while it should be categorical or binary.')
        return y
    
    def check_number_of_classes(self, y):
        """Check number of classes in the dataset and provide information whether it is binary or multiclassification problem.

        Args:
            y ([np.array], shape=(n_samples, )): Numpy array of provided target variable.

        Returns:
            number_of_classes ([int]): Number of classes in target variable.
        """
        number_of_classes = len(np.unique(y))
        return number_of_classes

    def check_features_and_target_names(self, features_names, target_name):
        """Checks what are the column names for input data and target data.

        Args:
            features_names ([list], shape=(self.X_train.shape[1],)): list of input data variable names.
            target_name ([list], shape=(1,)): list of input data target name.

        Returns:
            features_names ([list], shape=(self.X_train.shape[1],)): Features names of input data
            target_name ([list], shape=(1,)): Features names of input data
        """
        try:
            features_names = self.X_train.columns.tolist()
            target_name = self.y_train.columns[0]
        except: 
            if(features_names == None or target_name == None):
                raise ValueError('No feature names provided')
        return features_names, target_name

    def change_to_one_hot_encode(self, y):
        """Change 1-d vector to 2-d vector for target variable (simmilar to OneHotEncoding).

        Args:
            y ([np.array], shape=(n_samples, )): Numpy array of provided target variable.

        Returns:
            y_transformed ([np.array], shape=(n_samples, number_of_classes)): [description]
        """
        if(y.ndim == 2):
            y = y.squeeze()
        y_transformed = np.zeros((y.size, y.max()+1))
        y_transformed[np.arange(y.size), y] = 1
        return y_transformed

    def check_for_object_columns(self, X):
        """Check whether input vector contains any object data.

        Args:
            X ([np.array], shape=(n_samples, n_features)): Input vector.

        Returns:
            np.array(X) ([np.array], shape=(n_samples, n_features): Training vector.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.select_dtypes(include=np.number).shape[1] != X.shape[1]:
            raise TypeError('Your data contains object or string columns. Numeric data is obligated.')
        return np.array(X)
    
    def check_intercept(self, X, features_names):
        """Check whether there should be added intercept or not. 

        Args:
            X ([np.array], shape=(n_samples, n_features)): Input vector.
            features_names ([list], shape=(n_features,)): list of input data variable names.

        Returns:
        if(self.fit_intercept==True)
            np.array(X) ([np.array], shape=(n_samples, n_features+1): Training vector with intercept as first column.
            features_names ([list], shape=(n_features+1,)): list of input data variable names + "Intercept" as first element.
        else:
            np.array(X) ([np.array], shape=(n_samples, n_features): Training vector.
            features_names ([list], shape=(n_features,)): list of input data variable names.
        """
        if(self.fit_intercept==True):
            X = np.column_stack([np.ones(shape=(X.shape[0],)), X])
            features_names.insert(0, "Intercept")
        return X, features_names
    
    def initialize_weights(self):
        """Initialize coefficient weights.

        Returns:
            init_coef ([np.array], shape=(n_features, number_of_classes)): Matrix of random float values [0, 1] with given shape.
        """
        init_coef = np.random.random(size=(self.X_train.shape[1], self.number_of_classes-1))
        return init_coef
    
    def find_coefficients_gradient_descent(self, X, y, learning_rate, max_iter):
        """Perform gradient descent to find best coefficient values.

        Args:
            X ([np.array], shape=(n_samples, n_features)): Training vector.
            y ([np.array], shape=(n_samples, n_classes)): Numpy array of target variable.
            learning_rate ([float]): Specifies learning rate for gradient descent optimization.
            max_iter ([int]): Maximum number of iterations taken for the solvers to converge.
        """
        self.losses = np.zeros(shape=(max_iter, self.number_of_classes-1))
        for klasa in range(1, self.number_of_classes):
            current_coef = self.coef_[:, klasa-1]
            for epoch in range(0, max_iter):
                linear_model = self.calculate_linear_model(X=X, coef=current_coef)
                y_predict = self.calculate_sigmoid(linear_model=linear_model)
                grad_coef = self.derivative_of_loss(X=X, y=y[:,klasa], y_predict=y_predict)
                self.losses[epoch, klasa-1] = self.calculate_log_loss(y=y[:, klasa], y_predict=y_predict)
                current_coef = current_coef - learning_rate*grad_coef
            self.coef_[:, klasa-1] = current_coef
    
    def calculate_linear_model(self, X, coef):
        """Calculate simple linear model.

        Args:
            X ([np.array], shape=(n_samples, n_features)): Input vector.
            coef ([np.array], shape=(n_features+intercept,)): Coefficient for given class.

        Returns:
            np.matmul(coef, X.T) ([np.array], shape=(n_samples, )): linear model value: B_0+B_1*x_1+B_2*x_2+...+B_M*x_M
        """
        return np.matmul(coef, X.T)
    
    def calculate_sigmoid(self, linear_model):
        """Calculations of sigmoid function for current model.

        Args:
            linear_model ([np.array], shape=(n_samples, )): linear model value: B_0+B_1*x_1+B_2*x_2+...+B_M*x_M

        Returns:
            np.exp(linear_model)/(1+np.exp(linear_model)) ([np.array], shape=(n_samples, )): Value for sigmoid function of our model.
        """
        linear_model = np.where(linear_model > 700, 700, linear_model)
        return 1/(1+np.exp(-linear_model))

    def derivative_of_loss(self, X, y, y_predict):
        """Calculate derivative of loss with respect to coefficient.

        Args:
            X ([np.array], shape=(n_samples, n_features)): Input vector.
            y ([np.array], shape=(n_samples, n_classes)): Numpy array of target variable.
            y_predict ([np.array], shape=(n_samples, )): Numpy array of target variable for given class.

        Returns:
            (np.reshape(y_predict-y,(X.shape[0], 1))*X).mean(axis = 0) ([np.array], shape=(n_features+intercept, )): Gradient of coefficient.
        """
        return (np.reshape(y_predict-y,(X.shape[0], 1))*X).mean(axis = 0)
    
    def calculate_log_loss(self, y, y_predict):
        """Calculate log loss for current predictions. 

        Args:
            y ([np.array], shape=(n_samples, n_classes)): Numpy array of target variable.
            y_predict ([np.array], shape=(n_samples, )): Numpy array of target variable for given class.

        Returns:
            1/y.shape[0]*(y*np.log(y_predict, where=(y_predict>0))+(1-y)*np.log(1-y_predict, where=(1-y_predict>0))).mean() ([float]): Value of current log loss function.
        """
        return 1/y.shape[0]*(y*np.log(y_predict, where=(y_predict>0))+(1-y)*np.log(1-y_predict, where=(1-y_predict>0))).mean()
    
    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X ([array-like], shape=(n_predict_samples, n_features)): input data for predictions.

        Returns:
            np.argmax(probabilities_array, axis=1) ([np.array], shape=(n_predict_samples, )): Vector containing the class labels for each sample.
        """
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X, _ = self.check_intercept(X=X, features_names=[])
        probabilities_array = self.get_proba(X=X, coef=self.coef_)
        return np.argmax(probabilities_array, axis=1)
    
    def predict_proba(self, X):
        """Probability estimates.

        Args:
            X ([array-like], shape=(n_predict_samples, n_features)): input data for predictions.

        Returns:
            probabilities_array ([np.array], shape=(n_predict_samples, number_of_classes)): Probability of the sample for each class in the model.
        """
        self.check_fit(fit_used=self.fit_used)
        X = self.check_X(X=X, train=False)
        X = self.check_for_object_columns(X=X)
        X, _ = self.check_intercept(X=X, features_names=[])
        probabilities_array = self.get_proba(X=X, coef=self.coef_)
        return probabilities_array

    def check_fit(self, fit_used):
        """Check whether out object is fitted, if not then raise error.

        Args:
            fit_used ([bool]): Indicates whether our class has already been trained or not yet.
        """
        if fit_used == False:
            raise AttributeError('Logistic Regression has to be fitted first.')
    
    def get_proba(self, X, coef):
        """Estimates probability for given input and coefficient.

        Args:
            X ([np.array], shape=(n_samples, n_features)): Input vector.
            coef ([np.array], shape=(n_features+intercept, n_classes)): Coefficient.

        Returns:
            probabilities_array: ([np.array], shape=(n_predict_samples, number_of_classes)): Probability of the sample for each class in the model.
        """
        probabilities_array = np.zeros(shape=(X.shape[0], self.number_of_classes))
        all_linear_models = np.matmul(X, coef)
        sum_of_all_exp_classes = np.sum(np.exp(all_linear_models), axis=1)
        for klasa in range(1, self.number_of_classes):
            linear_model = np.matmul(X, coef[:,klasa-1])
            probabilities_array[:, klasa] = np.exp(linear_model)/(1+sum_of_all_exp_classes)
        probabilities_array[:, 0] = 1/(1+sum_of_all_exp_classes)
        return probabilities_array
    
    def summary(self):
        summary_frame=pd.DataFrame()
        summary_frame["Variables"] = self.features_names
        summary_frame["Coefficients"] = self.coef_
        return summary_frame