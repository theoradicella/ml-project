from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform, uniform


# Sampler configurations
sampler_configs = [
    {"sampler": [None]},
    # The floats in the strategies are the ratios of the minority class over the majority class
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
    # {"dim_reduction": [LDA()]},  # LDA is not compatible with the dataset (negative values when scaled)
    {
        # SFS starts with empty set of features and adds one by one
        # At each step it trains the model with the rest of the features and selects the best
        "dim_reduction": [SFS(estimator=Perceptron(), cv=None, scoring="f1")],
        "dim_reduction__estimator": [Perceptron(), LogisticRegression()],
        "dim_reduction__k_features": [3, 7, 15],  # Number of features to maintain
    },
]

# Classifier configurations
classifier_configs = [
    {
        "classifier": [Perceptron()],
        "classifier__max_iter": [100, 200],
        # loguniform is used to sample mostly small values but also a few large values
        # This is useful for hyperparameters that are usually small but can be large
        "classifier__eta0": loguniform(0.001, 10),
        # Handles potential class imbalances through weights
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier": [LogisticRegression(solver="lbfgs")],
        "classifier__max_iter": [700, 750],
        "classifier__C": uniform(1, 10),
        "classifier__penalty": ["l2"],  # Only L2 for lbfgs
        "classifier__class_weight": [None, "balanced"],
    },
    {
        "classifier": [KNeighborsClassifier()],
        "classifier__n_neighbors": [
            10,
            30,
            45,
            75,
        ],
    },
    {
        # This model outperforms the rest, it also takes a lot of time on the cv making the development of the project very slow.
        "classifier": [GradientBoostingClassifier()],
        "classifier__n_estimators": [300],  # Number of trees
        "classifier__max_depth": [5],  # Maximum depth of each tree
    },
]
