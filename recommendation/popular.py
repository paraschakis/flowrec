from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from collections import defaultdict
import numpy as np
from utils.shared_data import SharedData as Data


class PopularClassifier(BaseSKMObject, ClassifierMixin):
    """Recommender based on item popularity

    Notes
    ----------
    Can be made event-specific by providing the event type identifier

    Parameters
    ----------
    event_type: int (default=None)
        Event type for which popularity will be tracked. None for any event type.
    sliding_window: boolean (default=False)
        Whether to keep counts only within the sliding window
    """
    def __init__(self, event_type=None, sliding_window=False):
        super().__init__()
        self.event_type = event_type
        self.sliding_window = sliding_window
        self._rec_tracker = defaultdict(list)
        self._num_examples = 0

    def configure(self, **kwargs):
        self.counts = np.zeros(len(Data.classes))

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            for i in range(row_cnt):
                # decrement oldest item count if sliding window is enabled
                if self.sliding_window and self._num_examples > Data.window.max_size:
                    y_tail = Data.window.get_targets_matrix()[0]
                    y_tail_idx = np.searchsorted(Data.classes, y_tail)[0]
                    self.counts[y_tail_idx] -= 1  # forget observation
                if self.event_type is None or X[Data.eid] == self.event_type:
                    y_idx = np.searchsorted(Data.classes, y)[0]
                    self.counts[y_idx] += 1
                self._num_examples += 1

    def predict(self, X):
        predictions = []
        y_proba = self.predict_proba(X)
        r, _ = get_dimensions(X)
        for i in range(r):
            nonzero = np.nonzero(y_proba[i])[0]
            if len(nonzero > 0):
                sorted_desc = np.argsort(y_proba[i][nonzero])[::-1]
                sorted_ids = nonzero[sorted_desc]
                if not Data.allow_reminders:
                    sorted_ids = sorted_ids[~np.isin(sorted_ids, Data.session_vector)]
                if not Data.allow_repeated:
                    session = X[i, Data.sid]
                    sorted_ids = sorted_ids[~np.isin(sorted_ids, self._rec_tracker[session])]
                    self._rec_tracker[session].extend(sorted_ids[:Data.rec_size])
                y_pred = Data.classes[sorted_ids[:Data.rec_size]]
            else:
                y_pred = np.array([])
            predictions.append(y_pred)
        return np.array(predictions)

    def predict_proba(self, X):
        predictions = []
        r, _ = get_dimensions(X)
        for i in range(r):
            y_proba = np.zeros(len(Data.classes))
            nonzero = np.nonzero(self.counts)[0]
            if len(nonzero) > 0:
                y_proba[nonzero] = self.counts[nonzero] / max(self.counts[nonzero])
                y_proba[Data.session_vector[-1]] = 0.0
            predictions.append(y_proba)
        return np.array(predictions)
