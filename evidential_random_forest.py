"""
Evidential Random Forest used with imperfectly labeled data.

Author : Arthur Hoarau
Date : 05/10/2022
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from lib import decision_tree_imperfect
from lib import ibelief
import numpy as np
import math


class ERF(BaseEstimator, ClassifierMixin):
    """
    ERF for Evidential Random Forest, it is used to predict labels when input data
    are imperfectly labeled.
    """

    def __init__(self, n_estimators = 50, min_samples_leaf = 1, criterion = "conflict", rf_max_features="sqrt"):
        """
        ERF for Evidential Random Forest, it is used to predict labels when input data
        are imperfectly labeled.

        Parameters
        -----
        min_samples_leaf: int
            Minimum number of samples in a leaf
        criterion: string
            Usef criterion for splitting nodes. Use "conflict" for Jousselme distance + inclusion degree, "jousselme" for the Jousselme distance, 
            "euclidian" for euclidian distance, "uncertainty" for nons-pecificity + discord degree.

        Returns
        -----
        The instance of the class.
        """

        if (criterion not in ["euclidian", "conflict", "jousselme", "uncertainty"]):
            raise ValueError("Wrong selected criterion")

        # Used to retrieve the state of the model
        self._fitted = False

        # Parameters
        self.n_estimators = n_estimators

        # The decision trees estimators
        self.estimators = []
        for _ in range(n_estimators):
            self.estimators.append(decision_tree_imperfect.EDT(min_samples_leaf=min_samples_leaf, criterion=criterion, rf_max_features=rf_max_features))
        self.estimators = np.array(self.estimators)
    
    def score(self, X, y_true, criterion=3):
        """
        Calculate the accuracy score of the model,
        unsig a specific criterion in "Max Credibility", 
        "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions
        criterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        y_pred = self.predict(X, criterion=criterion)

        # Compare with true labels, and compute accuracy
        return accuracy_score(y_true, y_pred)
        pass

    def score_u65(self, X, y_true):
        """
        Calculate the u65 score of the model.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        _, y_pred = self.predict(X, return_bba=True)

        score = 0

        for i in range(X.shape[0]):
            bel = ibelief.mtobel(y_pred[i])
            pl = ibelief.mtopl(y_pred[i])
            
            if bel[1] >= 0.5:
                if 0 == y_true[i]:
                    score += 1
            elif pl[1] < 0.5:
                if 1 == y_true[i]:
                    score += 1
            else:
                score += (-1.2) * 0.5**2 + 2.2 * 0.5
            
        score = score / X.shape[0]

        print(score)
        input()
        # Compare with true labels, and compute accuracy
        return score

    def score_ssa(self, X, y_true):
        """
        Calculate the single set accuracy score of the model.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        _, y_pred = self.predict(X, return_bba=True)

        score = 0
        total = 0

        for i in range(X.shape[0]):
            
            bel = ibelief.mtobel(y_pred[i])
            pl = ibelief.mtopl(y_pred[i])
            
            if bel[1] >= 0.5:
                if 0 == y_true[i]:
                    score += 1
                total += 1
            elif pl[1] < 0.5:
                if 1 == y_true[i]:
                    score += 1
                total += 1
            
        score = score / total

        print(score, total)
        input()
        # Compare with true labels, and compute accuracy
        return score

    def score_Jouss(self, X, y_true):
        """
        Calculate the Jousselme distance score of the model.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        _, y_pred = self.predict(X, return_bba=True)

        score = 0

        D = ibelief.Dcalculus(y_pred.shape[1])
        for i in range(X.shape[0]):
            true_mass = np.zeros(y_pred.shape[1])
            true_mass[2**y_true[i].astype(int)] = 1
            score += ibelief.JousselmeDistance(y_pred[i], true_mass, D=D)


        score = score / X.shape[0]

        # Compare with true labels, and compute accuracy
        return 1 - score

    def get_estimators(self):
        """
        Returns the estimators.

        Returns
        -----
        ndarray :
            The array of EDT Evidential Decision Trees.
        """
        return self.estimators

    def predict_proba(self, X):
        """
        Predict class by returning pignistic probabilities

        Parameters
        -----
        X : ndarray
            Input array of X's

        Returns
        -----
        The pignistic probabilities for each class.
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        _, y_pred = self.predict(X, return_bba=True)

        predictions = ibelief.decisionDST(y_pred.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")
        
        # Predictions for each estimator
        bbas = []
        for estimator in self.estimators:
            _, predictions = estimator.predict(X, return_bba=True)
            bbas.append(predictions)
        bbas = np.array(bbas)

        # Aggregating
        result = np.zeros((X.shape[0], self.y_trained.shape[1]))
        for j in range(result.shape[0]):
            result[j] = ibelief.DST(bbas[:, j].T, criterion=12).T
            
        # Max Plausibility
        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        if return_bba:
            return predictions, result
        else:
            return predictions
        

    def fit(self, X, y):
        """
        Fit the model according to the training data.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y : ndarray
            Labels array

        Returns# Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1] + 1, 2).is_integer():
            y = np.hstack((np.zeros((y.shape[0],1)), y))
        elif not math.log(y.shape[1], 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")
        """

        # Check for data integrity
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        
        # Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1] + 1, 2).is_integer():
            y = np.hstack((np.zeros((y.shape[0],1)), y))
        elif not math.log(y.shape[1], 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        # Save X and y
        self.X_trained = X
        self.y_trained = y

        # Save size of the dataset
        self.size = self.X_trained.shape[0]

        # The model is now fitted
        self._fitted = True

        # Compute Bagging
        self.compute_bagging()


        return self

    def compute_bagging(self):
        """
        Computes Bagging.
        """

        # Bootstrap
        bootstrap_indices = self._bootstrap()

        # Fit Decision Trees
        self._fit_estimators(bootstrap_indices)
        
        # Aggregating
        # Aggregating is done during prediction phase: use score(), predict() or predict_proba() methods.


    def _fit_estimators(self, indices):
        """
        Fit every Decision Trees acconring to its bagging collection.

        Parameters
        -----
        indices: ndarray
            List of indices for each bagging collection.
        """
        for i in range(self.n_estimators):
            #print("Fitting Tree n*", i)
            self.estimators[i].fit(self.X_trained[indices[i]], self.y_trained[indices[i]])


    def _bootstrap(self):
        """
        Creates the bootstrap collection.

        Returns
        -----
        ndarray :
            Array of bootstrapped indices for each estomator.
        """
        bootstrap_indices = []

        # Bootstrap, randomly draws with replacement a new collection
        for _ in range(self.n_estimators):
            bootstrap_indices.append(np.random.choice(range(self.size), size=self.size))

        return np.array(bootstrap_indices)
        