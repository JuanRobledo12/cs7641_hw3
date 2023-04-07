import numpy as np
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose

        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        #print(X)
        #print(X.shape)
        mean_X_features = np.mean(X, axis=0)
        #print(mean_X_features)
        center_X = X - mean_X_features
        #print(center_X)
        self.U, self.S, self.V = np.linalg.svd(center_X, full_matrices=False)
        #print(self.U)
        #print(self.S)
        #print(self.V)


        return

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data

        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        
        #print(data)
        #print('U ---------\n', self.U)
        #print('S ---------\n', self.S)
        #print('V ---------\n', self.V)
        mean_data_features = np.mean(data, axis=0)
        center_data = data - mean_data_features
        X_new = center_data @ np.transpose(self.V)
        X_new = X_new[:,:K]
        #print(X_new)
        return X_new

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        
        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.

        """
        #print(retained_variance)
        mean_data_features = np.mean(data, axis=0)
        center_data = data - mean_data_features

        #print(self.S)
        #print(self.S**2)
        #print(np.cumsum(self.S**2))
        exp_var = np.cumsum(self.S**2)/ np.sum(self.S**2)
        #print(exp_var)
        calc_k = np.where(exp_var == exp_var[exp_var >= retained_variance][0])[0][0] + 1
        #print(calc_k)
        X_new = center_data @ np.transpose(self.V)[:,:calc_k]
        return X_new

    def get_V(self) -> np.ndarray:
        """ Getter function for value of V """

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig=None) -> None:  # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        Hint: Remember to label each of the plots when looping through. Refer to https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """
        self.fit(X)
        X_transform = self.transform(X, K=2)
        colors = ['blue', 'magenta', 'red']
        labels = np.unique(y)
        for label, color in zip(labels, colors):
            plt.scatter(X_transform[y == label, 0], X_transform[y == label, 1], c=color, label=label, marker='x')
        
        print('data before PCA', X.shape)
        print('data shape with PCA', X_transform.shape)
        plt.legend()
        plt.show()

        return

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()
