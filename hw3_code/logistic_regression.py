import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class LogisticRegression(object):
    def __init__(self):
        pass

    def sigmoid(self, s: np.ndarray) -> np.ndarray: # [2pts]
        """Sigmoid function 1 / (1 + e^{-s}).
        Args:
            s: (N, D) numpy array
        Return:
            (N, D) numpy array, whose values are transformed by sigmoid function to the range (0, 1)
        """
        sigmoid_arr = 1 / ((1 + np.exp(-1*s)))
        return sigmoid_arr

    def bias_augment(self, x: np.ndarray) -> np.ndarray: # [3pts]
        """Prepend a column of 1's to the x matrix

        Args:
            x (np.ndarray): (N, D) numpy array, N data points each with D features

        Returns:
            x_aug: (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
        """
        bias_arg = np.ones((x.shape[0], 1))
        x_aug = np.concatenate((bias_arg, x), axis=1)
        return x_aug
    

    def predict_probs(self, x_aug: np.ndarray, theta: np.ndarray) -> np.ndarray: # [5pts]
        """Given model weights theta and input data points x, calculate the logistic regression model's
        predicted probabilities for each point

        Args:
            x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
            theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model

        Returns:
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label
                this result is h(x) = P(y = 1 | x)
        """
        #print(theta)
        h_x = self.sigmoid(x_aug @ theta)
        return h_x

    def predict_labels(self, h_x: np.ndarray) -> np.ndarray: # [2pts]
        """Given model weights theta and input data points x, calculate the logistic regression model's
        predicted label for each point

        Args:
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of each data point being the positive label

        Returns:
            y_hat (np.ndarray): (N, 1) numpy array, the predicted labels of each data point
                0 for negative label, 1 for positive label
        """
        #print(h_x)
        y_hat = np.zeros_like(h_x)
        y_hat[h_x >= 0.5] = 1
        y_hat[h_x < 0.5] = 0
        #print(y_hat)
        return y_hat
        
        
        raise NotImplementedError

    def loss(self, y: np.ndarray, h_x: np.ndarray) -> float: # [3pts]
        """Given the true labels y and predicted probabilities h_x, calculate the
        binary cross-entropy loss

        Args:
            y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
            h_x (np.ndarray): (N, 1) numpy array, the predicted probabilities of being positive
        Return:
            loss (float)
        """
        loss_first_term = y * np.log(h_x)
        loss_second_term = (1 - y) * np.log(1 - h_x)
        loss = (-1/y.shape[0]) * np.sum(loss_first_term + loss_second_term)
        return loss

    def gradient(self, x_aug: np.ndarray, y: np.ndarray, h_x: np.ndarray) -> np.ndarray: # [3pts]
        """
        Calculate the gradient of the loss function with respect to the parameters theta.

        Args:
            x_aug (np.ndarray): (N, D + 1) numpy array, N data points each with a column of 1s and D features
            y (np.ndarray): (N, 1) numpy array, the true labels for each of the N points
            h_x: (N, 1) numpy array, the predicted probabilities of being positive
                    it is calculated as sigmoid(x multiply theta)

        Return:
            grad (np.ndarray): (D + 1, 1) numpy array,
                the gradient of the loss function with respect to the parameters theta.
        """
        grad =  (1 / x_aug.shape[0]) * (np.transpose(x_aug)@(h_x - y))
        return grad

    def accuracy(self, y: np.ndarray, y_hat: np.ndarray) -> float: # [2pts]
        """Calculate the accuracy of the predicted labels y_hat

        Args:
            y (np.ndarray): (N, 1) numpy array, true labels
            y_hat (np.ndarray): (N, 1) numpy array, predicted labels

        Return:
            accuracy of the given parameters theta on data x, y
        """
        true_val = y == y_hat
        true_val_count = true_val[true_val == True].size
        #print(true_val)
        #print(true_val_count)
        accuracy = true_val_count / y.shape[0]
        return accuracy

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> Tuple[float, float]: # [5pts]
        """Given data points x, labels y, and weights theta
        Calculate the loss and accuracy

        Don't forget to add the bias term to the input data x.

        Args:
            x (np.ndarray): (N, D) numpy array, N data points each with D features
            y (np.ndarray): (N, 1) numpy array, true labels
            theta (np.ndarray): (D + 1, 1) numpy array, the parameters of the logistic regression model

        Returns:
            Tuple[float, float]: loss, accuracy
        """
        x_aug = self.bias_augment(x)
        h_x = self.predict_probs(x_aug, theta)
        loss = self.loss(y,h_x)
        y_hat = self.predict_labels(h_x)
        accuracy = self.accuracy(y,y_hat)
        return loss, accuracy

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        lr: float,
        epochs: int,
    ) -> Tuple[
        np.ndarray, List[float], List[float], List[float], List[float], List[int]
    ]: # [5pts]
        """Use gradient descent to fit a logistic regression model

        Pseudocode:
        1) Initialize weights and bias `theta` with zeros
        2) Augment the training data for simplified multication with the `theta`
        3) For every epoch
            a) For each point in the training data, predict the probability h(x) = P(y = 1 | x)
            b) Calculate the gradient of the loss using predicted probabilities h(x)
            c) Update `theta` by "stepping" in the direction of the negative gradient, scaled by the learning rate.
            d) If the epoch = 0, 100, 200, ..., call the self.update_evaluation_lists function
        4) Return the trained `theta`

        Args:
            x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
            y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
            x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
            y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
            lr (float): Learning Rate
            epochs (int): Number of epochs (e.g. training loop iterations)
        Return:
            theta: (D + 1, 1) numpy array, the parameters of the fitted/trained model
        """
        theta = None
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.epoch_list = []

        ##########################
        ### START STUDENT CODE ###
        ##########################
        theta = np.zeros((x_train.shape[1] + 1, 1))
        x_train_aug = self.bias_augment(x_train)

        for epoch in range(epochs):
            h_x = self.predict_probs(x_train_aug, theta)
            gradient = self.gradient(x_train_aug, y_train, h_x)
            new_theta = theta - lr*gradient
            theta = np.copy(new_theta)
            if ((epoch % 100) == 0):
                self.update_evaluation_lists(x_train,y_train, x_val, y_val, theta, epoch)
        
        return new_theta

        ########################
        ### END STUDENT CODE ###
        ########################

    def update_evaluation_lists(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta: np.ndarray,
        epoch: int,
    ):
        """
        PROVIDED TO STUDENTS

        Updates lists of training loss, training accuracy, validation loss, and validation accuracy

        Args:
            x_train (np.ndarray): (N, D) numpy array, N training data points each with D features
            y_train (np.ndarray): (N, 1) numpy array, the true labels for each of the N training data points
            x_val (np.ndarray): (N, D) numpy array, N validation data points each with D features
            y_val (np.ndarray): (N, 1) numpy array, the true labels for each of the N validation data points
            theta: (D + 1, 1) numpy array, the current parameters of the model
            epoch (int): the current epoch number
        """

        train_loss, train_acc = self.evaluate(x_train, y_train, theta)
        val_loss, val_acc = self.evaluate(x_val, y_val, theta)
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.train_acc_list.append(train_acc)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch}:\n\ttrain loss: {round(train_loss, 3)}\ttrain acc: {round(train_acc, 3)}\n\tval loss:   {round(val_loss, 3)}\tval acc:   {round(val_acc, 3)}"
            )

    # provided function
    def plot_loss(
        self,
        train_loss_list: List[float] = None,
        val_loss_list: List[float] = None,
        epoch_list: List[int] = None,
    ) -> None:
        """
        PROVIDED TO STUDENTS

        Plot the loss of the train data and the loss of the test data.

        Args:
            train_loss_list: list of training losses from fit() function
            val_loss_list: list of validation losses from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_loss_list is None:
            assert hasattr(self, "train_loss_list")
            assert hasattr(self, "val_loss_list")
            assert hasattr(self, "epoch_list")
            train_loss_list = self.train_loss_list
            val_loss_list = self.val_loss_list
            epoch_list = self.epoch_list
        plt.clf()
        plt.plot(epoch_list, train_loss_list, label="train")
        plt.plot(epoch_list, val_loss_list, label="val")
        plt.title("Loss")
        plt.legend()
        plt.show()

    # provided function
    def plot_accuracy(
        self,
        train_acc_list: List[float] = None,
        val_acc_list: List[float] = None,
        epoch_list: List[int] = None,
    ) -> None:
        """
        PROVIDED TO STUDENTS

        Plot the accuracy of the train data and the accuracy of the test data.

        Args:
            train_loss_list: list of training losses from fit() function
            val_loss_list: list of validation losses from fit() function
            epoch_list: list of epochs at which the training and validation losses were evaluated

        Return:
            Do not return anything.
        """
        if train_acc_list is None:
            assert hasattr(self, "train_acc_list")
            assert hasattr(self, "val_acc_list")
            assert hasattr(self, "epoch_list")
            train_acc_list = self.train_acc_list
            val_acc_list = self.val_acc_list
            epoch_list = self.epoch_list
        plt.clf()
        plt.plot(epoch_list, train_acc_list, label="train")
        plt.plot(epoch_list, val_acc_list, label="val")
        plt.title("Accuracy")
        plt.legend()
        plt.show()
