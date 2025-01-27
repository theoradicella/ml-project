import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(
    estimator, X, y, ax, title="Learning Curve", cv=5, scoring="accuracy"
):
    """
    Plots a learning curve.

    Parameters:
        estimator: The model to evaluate.
        X: Feature set.
        y: Target labels.
        ax: Matplotlib axis to plot on.
        title: Title for the plot.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for evaluation.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    ax.plot(train_sizes, train_mean, label="Training Score", color="blue")
    ax.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange")

    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


def plot_roc_auc(y_test, y_prob, title="ROC Curve", ax=None):
    """
    Plots the ROC Curve and displays the AUC score.

    Parameters:
        y_test: Array-like, true labels for the test set.
        y_prob: Array-like, predicted probabilities for the positive class.
        title: Title for the plot.
        ax: Matplotlib axis to plot on (optional).
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    if ax is None:
        ax = plt.gca()  # Use the current axis if none is provided
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


def plot_precision_recall(y_test, y_prob, title="Precision-Recall Curve", ax=None):
    """
    Plots the Precision-Recall Curve and displays the AUC-PR score, along with a bisector line (m = -1).

    Parameters:
        y_test: Array-like, true labels for the test set.
        y_prob: Array-like, predicted probabilities for the positive class.
        title: Title for the plot.
        ax: Matplotlib axis to plot on (optional).
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)  # Calculate AUC-PR

    if ax is None:
        ax = plt.gca()  # Use the current axis if none is provided
    ax.plot(recall, precision, label=f"AUC = {auc_pr:.2f}")
    ax.plot([0, 1], [1, 0], linestyle="--", color="gray")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)


def plot_validation_curve(
    estimator,
    X,
    y,
    param_name,
    param_range,
    cv,
    scoring,
    ax=None,
    title="Validation Curve",
):
    """
    Plots the validation curve for a given estimator and hyperparameter on a specified axis.

    Parameters:
        estimator: The model to evaluate (e.g., GradientBoostingClassifier).
        X: Training data (features).
        y: Training data (labels).
        param_name: Name of the hyperparameter to vary (e.g., "n_estimators").
        param_range: Range of values for the hyperparameter.
        cv: Cross-validation splitting strategy.
        scoring: Scoring metric to evaluate the model (e.g., "f1").
        ax: Matplotlib axis to plot on (optional).
        title: Title for the plot.
    """
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Use provided axis or create one
    if ax is None:
        ax = plt.gca()

    # Plot the validation curve
    ax.plot(param_range, train_mean, label="Training Score", color="blue")
    ax.plot(param_range, test_mean, label="Validation Score", color="orange")
    ax.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue",
    )
    ax.fill_between(
        param_range,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color="orange",
    )

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


# Function to plot category-target relationships
def plot_category_target_count(df, column_name, target_column, target_value=1):
    """
    Plots the count of rows for each category in a given column where the target equals the target_value.

    Parameters:
    - df: pandas DataFrame
    - column_name: The name of the column to analyze (categorical feature).
    - target_column: The name of the target column.
    - target_value: The target value to filter on (default is 1).
    """
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the dataset.")
        return

    # Filter rows where target equals the target_value
    filtered_df = df[df[target_column] == target_value]

    # Group by the column and count rows
    category_counts = (
        filtered_df.groupby(column_name).size().sort_values(ascending=False)
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind="bar", color="skyblue")
    plt.title(f"Count of Rows by {column_name} where {target_column} = {target_value}")
    plt.ylabel("Count of Rows")
    plt.xlabel(column_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
