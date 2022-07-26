import math
from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


class BaseShiftDetector(ABC):
    @abstractmethod
    def is_drift(self, source_dataset, target_dataset):
        pass


class C2STShiftDetector(BaseShiftDetector):
    """
    Implements a detector based on Classifier Two-Sample Tests (C2ST),
    as shown in Lopez-Paz & Oquab, 2016: arXiv:1610.06545

    H_0: S = T --> p = p_trivial
    H_a: S != T --> p > p_trivial
    """
    #TODO: test would probably fail for heavily imbalanced datasets - raise warning/exception
    #TODO: implement streaming version as in Sequential Covariate Shift Detection Using Classifier Two-Sample Tests, Jang et al.

    def __init__(self, significance: float = 0.05):
        self.significance = significance

    def is_drift(self, source_dataset: np.ndarray, target_dataset: np.ndarray):
        source_dataset = np.hstack((source_dataset, np.array([[0] for _ in range(len(source_dataset))])))
        target_dataset = np.hstack((target_dataset, np.array([[1] for _ in range(len(target_dataset))])))
        dataset = np.concatenate([source_dataset, target_dataset], axis=0)
        # TODO: warning if not enough data
        X = dataset[:, :-1]
        y = dataset[:, -1]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, shuffle=True, stratify=y, random_state=42
        )

        t = DecisionTreeClassifier(random_state=42).fit(X_train, y_train).score(X_val, y_val)

        if len(X_val) < 30:
            raise NotImplementedError(
                "Not enough samples to approximate t distribution using central limit theorem."
            )

        # for large n, t follows a normal distribution under H0
        p_hat = DummyClassifier().fit(X_train, y_train).score(X_val, y_val)
        sigma = p_hat * (1.0 - p_hat) / len(y_val)

        p_value = 1 - 0.5 * (1 + math.erf((t - p_hat) / (sigma * math.sqrt(2))))

        return p_value < self.significance
