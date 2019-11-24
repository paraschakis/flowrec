from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from collections import deque
import numpy as np
from utils.shared_data import SharedData as data


class RandomClassifier(BaseSKMObject, ClassifierMixin):
    """Random recommender."""

    def __init__(self):
        super().__init__()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return

    def predict(self, X):
        predictions = deque()
        r, _ = get_dimensions(X)
        for i in range(r):
            y_prev = data.classes[data.session_vector[-1]]
            y_pred = [y_prev]
            while y_prev in y_pred:
                y_pred = np.random.choice(data.classes, data.rec_size)
            predictions.append(y_pred)
        return np.array(predictions)

    def predict_proba(self, X):
        """ Not implemented for this method."""
        raise NotImplementedError

    def __str__(self):
        return __class__.__name__
