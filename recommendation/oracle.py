from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from collections import deque
import numpy as np


class OracleClassifier(BaseSKMObject, ClassifierMixin):
    """Oracle recommender for testing purposes.

    Parameters
    ----------
    stream: Stream
        The stream from which to draw the samples.
    """

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return

    def predict(self, X):
        predictions = deque()
        r, _ = get_dimensions(X)
        y_pred = self.stream.current_sample_y
        for i in range(r):
            predictions.append(y_pred)
        return np.array(predictions)

    def predict_proba(self, X):
        """Not implemented for this method."""
        raise NotImplementedError
