from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import PCA, LatentDirichletAllocation as LDA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform


# Sampler configurations
sampler_configs = [
    {"sampler": [None]},
    # The floats in the strategies is the ratio of the minority class over the majority class
    {
        "sampler": [SMOTE()],  # KNN for oversampling
        "sampler__sampling_strategy": [0.5, 0.8, 1.0],
    },
    {
        "sampler": [RandomOverSampler()],  # Random oversampling by duplicating existing
        "sampler__sampling_strategy": [0.5, 0.8, 1.0],
    },
]

# Dimensionality reduction configurations
dim_reduction_configs = [
    {"dim_reduction": [None]},  # Bypass dimensionality reduction
    # Since 0 < n_components < 1, it represents the variance (deviation from the mean) for PCA
    {"dim_reduction": [PCA()], "dim_reduction__n_components": [0.5, 0.8, 0.9]},
    {"dim_reduction": [LDA()]},
    {
        # SFS starts with empty set of features and adds one by one
        # At each step it trains the model with the rest of the features and selects the best
        "dim_reduction": [SFS(estimator=Perceptron(), cv=None, scoring="f1")],
        "dim_reduction__estimator": [Perceptron(), LogisticRegression()],
        "dim_reduction__k_features": [5, 7, 10],  # Number of features to maintain
    },
]

# Classifier configurations
classifier_configs = [
    {
        "classifier": [Perceptron()],
        "classifier__max_iter": [1, 5, 10, 15, 50, 100],
        # loguniform is used to sample mostly small values but also a few large values
        # This is useful for hyperparameters that are usually small but can be large
        "classifier__eta0": loguniform(0.001, 100),
        # Handles potential class imbalances through weights
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier": [LogisticRegression(solver="saga")],
        "classifier__C": loguniform(0.001, 100),
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__n_estimators": [
            10,
            50,
            100,
            500,
        ],  # Number of trees in the random forest
    },
    {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [3, 5, 7, 9],  # Number of neighbors to consider
    },
]
