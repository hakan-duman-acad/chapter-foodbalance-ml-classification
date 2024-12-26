from memory_profiler import memory_usage
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import time
import numpy as np
import pickle


class Model:
    """
    A class to train and evaluate various machine learning models using cross-validation,
    hyperparameter tuning, and performance metrics.
    """

    def __init__(self, X, y, seed=2024, cv=5, model="RF"):
        """
        Initializes the Model class with the training data, target labels, random seed,
        cross-validation folds, and the model type to be used.

        Parameters:
        X (pd.DataFrame): Features of the training data.
        y (pd.Series): Target labels.
        seed (int): Random seed for reproducibility.
        cv (int): Number of cross-validation folds.
        model (str): The model type to use ("LR", "DT", "RF", "MLP").
        """
        self._X = X
        self._y = y
        self._cv = int(float(cv))
        self._seed = int(float(seed))
        self._models = [
            "LR",  # Logistic Regression
            "DT",  # Decision Tree
            "RF",  # Random Forest
            "MLP",  # Multi-layer Perceptron (Neural Network)
        ]
        # Ensure the provided model is valid, default to Random Forest if invalid
        if model not in self._models:
            self._model = self._models.index("RF")
        else:
            self._model = self._models.index(model)

        # Initialize additional attributes
        self.best_model = None
        self.best_params = None
        self._label_encoder = LabelEncoder().fit(self._y)

        # Define hyperparameter grids for each model
        self._pram_grid = [
            {
                "classifier": [LogisticRegression(max_iter=500)],
                "classifier__C": np.logspace(-6, 3, 256),
            },
            {
                "classifier": [DecisionTreeClassifier()],
                "classifier__max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "classifier__min_samples_split": [2, 5, 10, 15],
                "classifier__min_samples_leaf": [1, 3, 7, 11],
            },
            {
                "classifier": [RandomForestClassifier()],
                "classifier__n_estimators": [100, 200, 500, 1000],
                "classifier__max_depth": [5, 10, 15, 20],
                "classifier__min_samples_split": [2, 5, 10, 15],
                "classifier__min_samples_leaf": [1, 3, 7, 11],
            },
            {
                "classifier": [MLPClassifier(max_iter=500)],
                "classifier__hidden_layer_sizes": [(4,), (16,), (32,), (64,)],
                "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01],
                "classifier__activation": ["relu", "tanh"],
                "classifier__learning_rate": ["constant", "adaptive"],
            },
        ]
        self._model_grid = self._pram_grid[self._model]

    def train(self):
        """
        Trains the model using GridSearchCV with cross-validation and memory profiling.
        This method selects the best model based on the highest F1 score.
        """
        start = time.time()  # Start the timer for training

        y = self._label_encoder.transform(self._y)  # Encode the target labels
        folds = StratifiedKFold(
            n_splits=self._cv, shuffle=True, random_state=self._seed
        )  # Cross-validation folds
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standard scaling of features
                (
                    "classifier",
                    RandomForestClassifier(),
                ),  # Default classifier: Random Forest
            ]
        )

        # Define F1 scorer for GridSearchCV
        f1_scorer = make_scorer(f1_score, average="weighted")

        # Track memory usage during training
        m_usage = memory_usage(-1, interval=0.1)

        # Perform GridSearchCV to find the best hyperparameters
        best_model_gcv = GridSearchCV(
            pipeline, self._model_grid, scoring=f1_scorer, cv=folds, n_jobs=-1
        ).fit(self._X, y)

        # Store the best model and its parameters
        self.best_model = best_model_gcv.best_estimator_
        self.best_params = best_model_gcv.best_params_

        end = time.time()  # End the timer
        self._peak_memory = max(m_usage)  # Track peak memory usage
        self._training_time = end - start  # Track training time

    def predict(self, X=None):
        """
        Makes predictions using the trained model.

        Parameters:
        X (pd.DataFrame): Input features for prediction. If None, uses the training data.

        Returns:
        y_pred (np.ndarray): Predicted target labels.
        """
        if X is None:
            X = self._X  # Use training data if no input data is provided
        y_pred = self.best_model.predict(X)  # Predict using the best model
        y_pred = self._label_encoder.inverse_transform(
            y_pred
        )  # Decode predicted labels back to original values
        return y_pred

    def train_metrics(self):
        """
        Computes and stores classification metrics (precision, recall, F1-score) for the training set.
        """
        y_pred = self.predict()  # Get predictions for the training data
        self._train_metrics = classification_report(
            self._y, y_pred
        )  # Store classification metrics

    def test_metrics(self, X_test, y_test):
        """
        Computes and stores classification metrics (precision, recall, F1-score) for the test set.

        Parameters:
        X_test (pd.DataFrame): Features of the test data.
        y_test (pd.Series): True labels of the test data.
        """
        y_pred = self.predict(X=X_test)  # Get predictions for the test data
        self._test_metrics = classification_report(
            y_test, y_pred
        )  # Store classification metrics

    def get_train_metrics(self):
        """
        Returns the classification metrics for the training set.
        """
        self.train_metrics()  # Compute training metrics if not already done
        return self._train_metrics

    def get_test_metrics(self):
        """
        Returns the classification metrics for the test set.
        """
        return self._test_metrics

    def train_f1_score(self):
        """
        Computes the F1-score for the training set.
        """
        y_pred = self.predict()  # Get predictions for the training data
        return f1_score(
            self._y, y_pred, average="weighted"
        )  # Compute and return the weighted F1-score

    def test_f1_score(self, X_test, y_test):
        """
        Computes the F1-score for the test set.

        Parameters:
        X_test (pd.DataFrame): Features of the test data.
        y_test (pd.Series): True labels of the test data.
        """
        y_pred = self.predict(X=X_test)  # Get predictions for the test data
        return f1_score(
            y_test, y_pred, average="weighted"
        )  # Compute and return the weighted F1-score

    def get_time(self):
        """
        Returns the training time.
        """
        return self._training_time

    def get_peak_memory(self):
        """
        Returns the peak memory usage during training.
        """
        return self._peak_memory

    def get_all_info(self, X_test, y_test):
        """
        Returns all relevant performance information: training time, peak memory, F1-scores for train and test sets.

        Parameters:
        X_test (pd.DataFrame): Features of the test data.
        y_test (pd.Series): True labels of the test data.
        """
        self._train_f1_score = self.train_f1_score()  # Compute training F1-score
        self._test_f1_score = self.test_f1_score(
            X_test, y_test
        )  # Compute test F1-score
        return (
            self._training_time,
            self._peak_memory,
            self._train_f1_score,
            self._test_f1_score,
        )

    def get_best_params(self):
        """
        Returns the best hyperparameters found during training.
        """
        return self.best_params.copy()

    def save_model(self, path):
        """
        Saves the trained model to a specified path using pickle.

        Parameters:
        path (str): The path where the model should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)

    def load_model(self, path):
        """
        Loads a pre-trained model from a specified path using pickle.

        Parameters:
        path (str): The path from which to load the model.
        """
        with open(path, "rb") as f:
            self.best_model = pickle.load(f)


class Metrics:
    """
    A class to compute common classification metrics: precision, recall, and F1-score.
    """

    def __init__(self, y_observed, y_pred):
        """
        Initializes the Metrics class with observed and predicted labels.

        Parameters:
        y_observed (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        """
        self.y_observed = y_observed  # True labels
        self.y_pred = y_pred  # Predicted labels

    def precision(self):
        """
        Computes and returns the weighted precision score.
        """
        return precision_score(self.y_observed, self.y_pred, average="weighted")

    def recall(self):
        """
        Computes and returns the weighted recall score.
        """
        return recall_score(self.y_observed, self.y_pred, average="weighted")

    def f1_score(self):
        """
        Computes and returns the weighted F1 score.
        """
        return f1_score(self.y_observed, self.y_pred, average="weighted")
