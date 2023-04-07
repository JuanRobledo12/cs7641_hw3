import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        rmse_val = np.sqrt(np.sum((label - pred)**2) / pred.size)
        #print(rmse_val)
        return rmse_val

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        #print(x)
        #print(degree)
        degree_arr= np.arange(degree + 1)
        #print(degree_arr)

        if x.ndim > 1:
            degree_arr= degree_arr.reshape(1,-1, 1) #-1 lets np decide what n dimensions to add in that axis
            #print(degree_arr)
            #Create a new dimension for each row in x, equivalent to x = x[:, np.newaxis, :]
            x = np.expand_dims(x, axis=1)
            #print(x)
            feat = np.power(x, degree_arr)
            #print(feat)
        else:
            x = x.reshape((x.shape[0], 1))
            feat = x ** degree_arr
            #print(feat)
        

        return feat
    

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        #print(xtest)
        #print(weight)

        prediction = np.sum(xtest * np.transpose(weight), axis=1)
        prediction = prediction.reshape((prediction.shape[0], 1))
        #print(prediction)
        return prediction

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        
        #Equation at slide 17 of Linear regression PPT
        
        #print(xtrain.shape)
        #print(ytrain.shape)
        weight = np.linalg.pinv(xtrain)@ytrain
        #print(weight.shape)
        
        return weight

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1], 1))
        N = xtrain.shape[0]
        loss_per_epoch = []

        for epoch in range(epochs):

            new_weight =  weight + (learning_rate/N)*(np.transpose(xtrain)@(ytrain-(xtrain@weight)))
            ypred = self.predict(xtrain,new_weight)
            loss_per_epoch.append(self.rmse(ypred, ytrain))
            #print(new_weight)
            weight = np.copy(new_weight)
        
        return weight, loss_per_epoch

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1], 1))
        N = xtrain.shape[0]
        loss_per_epoch = []

        #print('epochs', epochs)
        #print('N', N)
        for epoch in range(epochs):
            for i in range(N):
                x_i = xtrain[i].reshape((1,xtrain.shape[1]))
                y_i = ytrain[i]
                new_weight = weight + ((learning_rate)*(np.transpose(x_i)@(y_i - (x_i@weight))))
                ypred = self.predict(xtrain,new_weight)
                loss_per_epoch.append(self.rmse(ypred, ytrain))
                weight = np.copy(new_weight)
        
        #print(weight)
        #print(loss_per_epoch)

        return weight, loss_per_epoch


            
        
        raise NotImplementedError

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        #Equation at slide 27 of regularized regression PPT
        
        z_t_z = np.transpose(xtrain)@xtrain
        #print(z_t_z)
        lamda_identity_matrix = c_lambda * np.identity(n=z_t_z.shape[0])
        #print(lamda_identity_matrix)
        lamda_identity_matrix[0,:] = lamda_identity_matrix[0,:] * 0
        #print(lamda_identity_matrix)
        z_lambda = z_t_z + lamda_identity_matrix
        #print(z_lambda)

        z_lambda_inv = np.linalg.pinv(z_lambda)
        
        #print(z_lambda_inv)
        reg_weight = z_lambda_inv @ (np.transpose(xtrain)@ytrain)
        #print(reg_weight)
        return reg_weight
        
    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1], 1))
        N = xtrain.shape[0]
        loss_per_epoch = []

        for epoch in range(epochs):
            
            #ypred = self.predict(xtrain, weight)
            gradient = ((-1 * np.transpose(xtrain)@(ytrain - xtrain@weight)) + c_lambda * weight) / N
            weight =  weight - (learning_rate * gradient)
            ypred = self.predict(xtrain, weight)
            loss_per_epoch.append(self.rmse(ypred, ytrain))
            
        #print(weight)
        
        return weight, loss_per_epoch
        

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1], 1))
        N = xtrain.shape[0]
        loss_per_epoch = []

        #print('epochs', epochs)
        #print('N', N)
        for epoch in range(epochs):
            for i in range(N):
                x_i = xtrain[i].reshape((1,xtrain.shape[1]))
                y_i = ytrain[i]
                gradient =  np.transpose(-x_i*(y_i - (x_i @ weight))) + ((c_lambda * weight)/N)
                #print(gradient)
                weight =  weight - (learning_rate * gradient)
                ypred = self.predict(xtrain, weight)
                loss_per_epoch.append(self.rmse(ypred, ytrain))
            
        #print(weight)
        #print(loss_per_epoch)

        return weight, loss_per_epoch


    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> List[float]:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each kfold

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        
        #print(X.shape)
        #print(kfold)
        X_list = np.array_split(X, kfold, axis=0)
        y_list = np.array_split(y, kfold, axis=0)
        #print('Datapoints -----------\n', X_list)
        loss_per_fold = []

        for k in range(kfold):
            #Create the kth test set
            X_kfold_test = X_list[k]
            y_kfold_test = y_list[k]

            X_list_cp = X_list.copy()
            y_list_cp = y_list.copy()

            X_list_cp.pop(k)
            y_list_cp.pop(k)
            
            #print('X_list_copy after pop', X_list_cp)
            X_kfold_train = np.concatenate(X_list_cp, axis=0)
            y_kfold_train = np.concatenate(y_list_cp, axis=0)

            #print('train data ----------\n',X_kfold_train)
            weight_arr = self.ridge_fit_closed(X_kfold_train, y_kfold_train, c_lambda)
            y_predicted = self.predict(X_kfold_test, weight_arr)
            rmse_loss = self.rmse(y_predicted, y_kfold_test)
            loss_per_fold.append(rmse_loss)
        #print(loss_per_fold)
        return loss_per_fold

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        FUNCTION PROVIDED TO STUDENTS
        
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm

        return best_lambda, best_error, error_list

