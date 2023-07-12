from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from requirements import *

class KNearestNeighbors():
    def __init__(self, n_neighbors=5, distance="euclidean", random_state=17):
        """Initialize K-Nearest Neighbors class.

        Args:
            n_neighbors ([int], optional): Number of neighbors simmilar to given observation. Defaults to 5.
            distance ([str], optional): Type of distance used in comparing observations. Defaults to "euclidean".
            random_state ([int], optional): Seed. Defaults to 17.
        """
        self.n_neighbors = n_neighbors
        distances = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_similarity
        }
        self.distance = distances[distance]
        self.random_state = random_state
        random.seed(self.random_state)

    def fit(self, X, y):
        """This function has to just remember these two arrays and decide what type of problem is there: classification or regression.

        Args:
            X ([array-like], shape=(n_samples, number_of_features)): Training input vector.
            y ([array-like], shape=(n_samples,)): Training target vector.
        """
        self.X_train = self.check_X(X=X, train=True)
        self.y_train = self.check_y(y=y, train=True)
    
    def check_X(self, X, train):
        """Check type of input data and raise errors is something is wrong with it.
        
        Args:
            X ([array-like], shape=(n_samples, number_of_features)): Input vector.
            train ([bool]): True if this is array given in fit() method and False if this was argument for predict.

        Returns:
            np.array(X) ([np.array], shape=(n_samples, number_of_features)): numpy array of provided input vector.
        """
        if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray) and not torch.is_tensor(X):
            raise TypeError('Wrong type of X. It should be dataframe, numpy array or torch tensor.')
        if(train == True):
            if(X.shape[0] < self.n_neighbors):
                raise ValueError(f"Expected n_neighbors <= n_samples, but n_samples = {X.shape[0]}, n_neighbors = {self.n_neighbors}")
        if(train == False):
            if(self.X_train.shape[1] != X.shape[1]):
                raise ValueError(f"X has {X.shape[1]} features, but KNeighbors is expecting {self.X_train.shape[1]} features as input.")
        return np.array(X)
    
    def check_y(self, y, train):
        """Check type of y and determine wether this is a classification or regression problem.

        Args:
            y ([array-like], shape=(n_samples,)): Target vector.
            train ([bool]): True if this is array given in fit() method and False if this was argument for predict.

        Returns:
            np.array(y) ([np.array], shape=(n_samples,)) numpy array of provided target vector.
        """
        if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series) and not isinstance(y, np.ndarray) and not torch.is_tensor(y):
            raise TypeError('Wrong type of y. It should be pandas DataFrame, pandas Series, numpy array or torch tensor.')
        if(train == True):
            if(len(np.unique(y)) > 30 or (self.X_train.shape[0] < 30 and len(np.unique(y)) == self.X_train.shape[0])):
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"
        return np.array(y)
    
    def predict(self, X):
        """Predict class labels for provided input vector.

        Args:
            X ([array-like], shape=(n_samples, number_of_features)): Provided input vector.

        Returns:
            predictions ([np.array], shape=(n_samples,)) predictions for provided input vector.
        """
        self.X_test = self.check_X(X=X, train=False)
        distances = self.distance(X_1=self.X_test, X_2=self.X_train)
        k_closest_indices = self.find_k_closest_indices(distances=distances)
        predictions = self.get_prediction(k_closest_indices=k_closest_indices)
        return predictions

    def predict_proba(self, X):
        """Predict probability of class labels for provided input vector.

        Args:
            X ([array-like], shape=(n_samples, number_of_features)): Provided input vector.
        
        Returns:
            probabilities ([np.array], shape=(n_samples, number_of_classes)) probabilities of each class for provided input vector.
        """
        self.X_test = self.check_X(X=X, train=False)
        distances = self.distance(X_1=self.X_test, X_2=self.X_train)
        k_closest_indices = self.find_k_closest_indices(distances=distances)
        probabilities = self.get_probabilities(k_closest_indices=k_closest_indices)
        return probabilities

    def euclidean_distance(self, X_1, X_2):
        """
        Calculate euclidean distance between two arrays.

        Args:
            X_1 ([np.array], shape=(n_samples_test_data, number_of_features)): X_test data.
            X_2 ([np.array], shape=(n_samples_train_data, number_of_features)): X_train data that we want to compare with.
        
        Returns:
            np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2)) ([np.array], shape=(n_samples_test_data, n_samples_train_data)): Euclidean distance between each observation from the test set and each observation from the training set.
        """
        return np.sqrt(np.sum((X_1[:,np.newaxis]-X_2)**2, axis=2))
    
    def manhattan_distance(self, X_1, X_2):
        """
        Calculate manhattan distance between two arrays.

        Args:
            X_1 ([np.array], shape=(n_samples_test_data, number_of_features)): X_test data.
            X_2 ([np.array], shape=(n_samples_train_data, number_of_features)): X_train data that we want to compare with.
        
        Returns:
            np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2) ([np.array], shape=(n_samples_test_data, n_samples_train_data)): Manhattan distance between each observation from the test set and each observation from the training set.
        """
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)
    
    def cosine_similarity(self, X_1, X_2):
        """
        Calculate cosine similarity between two arrays.

        Args:
            X_1 ([np.array], shape=(n_samples_test_data, number_of_features)): X_test data.
            X_2 ([np.array], shape=(n_samples_train_data, number_of_features)): X_train data that we want to compare with.
        
        Returns:
            np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)/(np.sum(X_1[:,np.newaxis])*np.sum(X_2)) ([np.array], shape=(n_samples_test_data, n_samples_train_data)): Cosine distance between each observation from the test set and each observation from the training set.
        """
        return np.sum(np.abs(X_1[:,np.newaxis]-X_2), axis=2)/(np.sum(X_1[:,np.newaxis])*np.sum(X_2))
    
    def cosine_similarity2(self, X_1, X_2):
        """
        Calculate cosine similarity between two arrays.

        Args:
            X_1 ([np.array], shape=(n_samples_test_data, number_of_features)): X_test data.
            X_2 ([np.array], shape=(n_samples_train_data, number_of_features)): X_train data that we want to compare with.
        
        Returns:
            np.sum(X_1[:,np.newaxis]*X_2, axis=2)/np.sqrt((np.sum(X_1[:,np.newaxis]**2)*np.sum(X_2**2))) ([np.array], shape=(n_samples_test_data, n_samples_train_data)): Cosine2 distance between each observation from the test set and each observation from the training set.
        """
        return np.sum(X_1[:,np.newaxis]*X_2, axis=2)/np.sqrt((np.sum(X_1[:,np.newaxis]**2)*np.sum(X_2**2)))
    
    def find_k_closest_indices(self, distances):
        """Find indices of nearest n_neighbors for each observation based on provided distance array.

        Args:
            distances ([np.array], shape=(n_samples_test_data, n_samples_train_data)): Distances between each test observation and training data points.
        
        Returns:
            np.argsort(distances)[:,:self.n_neighbors] ([np.array], shape=(n_samples_test_data, self.n_neighbors)): self.n_neighbors closest indices in training data for each observations.
        """
        return np.argsort(distances)[:,:self.n_neighbors]

    def get_prediction(self, k_closest_indices):
        """Function that returns predicted output. For classification most frequent class in first n_neighbors observations (that in terms of distance are most simmilar to test observation).
        For regression it will output weighted average of first num_neighbor observations.

        Args:
            k_closest_indices ([np.array], shape=(n_samples_test_data, self.n_neighbors)): self.n_neighbors closest indices in training data for each observations.
        
        Returns:
            Regression: np.average(target_values, axis=1, weights=[i for i in range(target_values.shape[1], 0, -1)]).squeeze(), ([np.array], shape=(n_samples_test_data,)): weighted average of closest indices.
            Classification: classification_task ([function]): redirects to classification_task function.
        """
        target_values = self.y_train[k_closest_indices]
        if(self.problem_type == "regression"):
            return np.average(target_values, axis=1, weights=[i for i in range(target_values.shape[1], 0, -1)]).squeeze()
        else:
            return self.classification_task(target_values=target_values)
        
    def classification_task(self, target_values):
        """Returns predictions.

        Args:
            target_values ([np.array], shape=(n_samples_test_data, self.n_neighbors)): Array of target train vector values with most simmilar class observations.
        
        Returns:
            chosen_values.squeeze() ([np.array], shape=(n_samples_test_data,)): Predictions for provided (in predict() method) input vector.
        """
        chosen_values = stats.mode(target_values, axis=1, keepdims=True)[0]
        number_of_occurences = stats.mode(target_values, axis=1, keepdims=True)[1].squeeze()
        #Just in case if we will have same count of classes in these most simmilar observations, then actually the most simmilar target will be returned 
        indices_to_be_replaced = np.where(number_of_occurences == self.n_neighbors/len(np.unique(self.y_train)))[0]
        np.put(chosen_values, indices_to_be_replaced, target_values[indices_to_be_replaced,0])
        return chosen_values.squeeze()

    def get_probabilities(self, k_closest_indices):
        """Function that returns probabilities output.

        Args:
            k_closest_indices ([np.array], shape=(n_samples_test_data, self.n_neighbors)): self.n_neighbors closest indices in training data for each observations.
        
        Returns:
            probability_task ([function]): redirects to probability_task function.
        """
        target_values = self.y_train[k_closest_indices]
        if(self.problem_type == "regression"):
            raise TypeError("predict_proba does not work for regression tasks.")
        else:
            return self.probability_task(target_values=target_values)
    
    def probability_task(self, target_values):
        """Returns probabilities of each class.

        Args:
            target_values ([np.array], shape=(n_samples_test_data, self.n_neighbors)): Array of target train vector values with most simmilar class observations.
        
        Returns:
            probabilities_array ([np.array], shape=(n_samples_test_data, number_of_classes)): Probabilities for provided (in predict_proba() method) input vector.
        """
        probabilities_array = np.zeros((self.X_test.shape[0], 0), dtype=float)
        for klasa in np.unique(self.y_train):
            probabilities_array = np.column_stack([probabilities_array, (target_values==klasa).sum(axis=1)/target_values.shape[1]]) 
        probabilities_array = probabilities_array.reshape(self.X_test.shape[0], len(np.unique(self.y_train)))
        return probabilities_array