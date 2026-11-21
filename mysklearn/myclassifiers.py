import numpy as np
from mysklearn import myutils

class MySimpleLinearRegressor:
    """
    
    A simple linear regression using the least squares method.
    
    Equation:
        y = slope * x + intercept
    
    Attributes:
        slope (float): Slope (m) of the line.
        intercept (float): Intercept (b) of the line.
    
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, slope=None, intercept=None):
        
        """
        Initialize the regressor with optional slope and intercept.
        If not provided, they will be computed with fit().
        
        Args:
            slope (float): The slope (m) of the line. None if to be computed.
            intercept (float): The intercept (b) of the line. None if to be computed.
     
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """
         Fits a simple linear regression line to X_train and y_train

        Args:
            X_train (list of list of numeric values): Each inner list represents a single sample with one feature.
            Shape: (n_train_samples, n_features)
            Example for 4 samples: [[1], [2], [3], [4]]

            y_train (list of numeric values): Target values corresponding to each sample in X_train.
            Shape: (n_train_samples,)
            Example: [2, 4, 6, 8]

        Notes:
        - In simple linear regression, each sample has only one feature (n_features = 1).
        - Think of X_train as a column of input values and y_train as a column of output values.
        - Using a "list of list" structure keeps the interface consistent with multi-feature datasets.

        """
        X_train = [x[0] for x in X_train]
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """
        Predict values for test data.
        
        Args:
            X_test (list of list of numeric vals): Test samples. Shape = (n_test_samples, 1).
        
        Returns:
            list of numeric vals: Predicted target values.
        Raises:
            ValueError: If model is not yet fitted (slope or intercept is None)
        
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        
        return [self.slope * x[0] + self.intercept for x in X_test]

    @staticmethod
    def compute_slope_intercept(x, y):
        """
        Compute slope (m) and intercept (b) using least squares.
        This is strictly for **univariate regression** (1 predictor).

        Args:
            x (list of numeric vals): Feature values (1D list).
            y (list of numeric vals): Target values (1D list).
        
        Returns: 
            tuple:
                m (float): Slope of the line.
                b (float): Intercept of the line.

        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) \
            / sum((x[i] - mean_x) ** 2 for i in range(len(x)))

        b = mean_y - m * mean_x
        return m, b

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = self.regressor.predict(X_test)

        y_predicted = [] 
        for prediction in predictions: 
            labels = self.discretizer(prediction)
            y_predicted.append(labels)

        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        distances = []
        neighbors = []
        
        for point_to_test in X_test:
            distances.append([]) 
            neighbors.append([])
            x_y_pairs = []
            for index in range(len(self.X_train)):
                point_to_train = self.X_train[index]
                distance = myutils.compute_euclidean_distance(point_to_test, point_to_train)

                x_y_pairs.append([distance,index])

            x_y_pairs.sort(key=lambda pair: pair[0])


            for distance, index in x_y_pairs[:self.n_neighbors]:
                distances[-1].append(distance)
                neighbors[-1].append(index) 
        
        return distances, neighbors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbors = self.kneighbors(X_test)
        y_predicted = []

        for neighbor in neighbors:
            labels = {}

            for index in neighbor:
                label = self.y_train[index]

                if label not in labels:
                    labels[label] = 0
                labels[label] += 1

            highest_count = 0
            most_common_label = None

            for label in labels:
                if labels[label] > highest_count:
                    highest_count = labels[label]
                    most_common_label = label

            y_predicted.append(most_common_label)
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        labels = {}
        highest_count = 0
        for label in y_train: 
            if label not in labels:  
                labels[label] = 0
            labels[label] += 1
        for label in labels:
            if labels[label] > highest_count:
                highest_count = labels[label]
                self.most_common_label = label

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []

        for point_to_test in X_test:
            y_pred.append(self.most_common_label)

        return y_pred

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return []
