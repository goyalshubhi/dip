"""Classical classifiers on the 22 features, tuned with GroupKFold over images."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_NAMES
from .utils import CLASSES, SEED, assert_not_test

N_FOLDS = 5


def model_grid() -> dict[str, tuple[Pipeline, dict]]:
    """Name -> (pipeline, hyperparameter grid). class_weight='balanced' where supported."""
    return {
        "GaussianNB": (
            Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11, 15, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "SVM-RBF": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=SEED))]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.03, 0.1]},
        ),
        "RandomForest": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1))]),
            {"clf__n_estimators": [300], "clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 3, 5]},
        ),
    }


def tune_and_fit(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, image_ids, jobs: int = -1):
    """GridSearchCV over GroupKFold(image) with macro F1; refit on all given (non-test) data."""
    assert_not_test(set(image_ids), f"tune_and_fit({name})")
    pipe, grid = model_grid()[name]
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=GroupKFold(n_splits=N_FOLDS), n_jobs=jobs, refit=True)
    search.fit(X, y, groups=groups)
    return search


def metrics(y_true, y_pred, labels=CLASSES) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, rc, _, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"n": int(len(y_true)), "accuracy": accuracy_score(y_true, y_pred),
           "macro_precision": p, "macro_recall": r, "macro_f1": f}
    for c, rec, sup in zip(labels, rc, support):
        out[f"recall_{c}"] = rec
        out[f"n_{c}"] = int(sup)
    return out


def confusion(y_true, y_pred, labels=CLASSES) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURE_NAMES].to_numpy(np.float64)
    assert np.isfinite(X).all(), "non-finite feature values"
    return X, df["label"].to_numpy(), df["image"].to_numpy()
