import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from scipy.sparse import lil_matrix
from collections import defaultdict
from utils.shared_data import SharedData as Data


class CoEventsClassifier(BaseSKMObject, ClassifierMixin):
    """Based on co-occurrence of two events of a given type in the same session.

    Notes
    ----------
    Corresponds to bigram association rules.
    Can be made event-specific (e.g. click <-> click) by providing the event type identifier

    References
    ----------
    Ludewig et al. (2018). In User Modeling and User-Adapted Interaction, 28(4-5), 331-390.
    "Evaluation of Session-based Recommendation Algorithms"

    Parameters
    ----------
    target_event_type: int (default=None)
        The event type id for creating item associations. None means any type.
    sliding_window: boolean (default=False)
        Whether or not to keep associations only for the events of the sliding window
    """

    def __init__(self, target_event_type=None, sliding_window=False):
        super().__init__()
        self.target_event_type = target_event_type
        self.sliding_window = sliding_window
        self._item_tracker = defaultdict(set)
        self._rec_tracker = defaultdict(list)
        self._num_examples = 0

    def configure(self, **kwargs):
        self.matrix = lil_matrix((len(Data.classes), len(Data.classes)), dtype=int)

    def update_matrix(self, row, col, value):
        self.matrix[row, col] += value
        self.matrix[col, row] += value

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            for i in range(row_cnt):
                self._partial_fit(X[i], y[i])

    def _partial_fit(self, X, y):
        y_idx = np.searchsorted(Data.classes, y)
        if self.sliding_window and self._num_examples > Data.window.max_size:
            self._remove_oldest_associations()
        self._num_examples += 1
        session = X[Data.sid]
        if self.target_event_type is None or self.target_event_type == X[Data.eid]:
            X_slice, y_slice = Data.window.get_slice(session, Data.sid)
            for i in range(len(X_slice)):
                if self.target_event_type is None or self.target_event_type == X_slice[i][Data.eid]:
                    y_o_idx = np.searchsorted(Data.classes, y_slice[i][0])
                    self.update_matrix(y_o_idx, y_idx, 1)

    def _remove_oldest_associations(self):
        x_tail = Data.window.get_attributes_matrix()[0]
        y_tail = Data.window.get_targets_matrix()[0]
        session_tail = x_tail[Data.sid]
        y_tail_idx = np.searchsorted(Data.classes, y_tail)[0]
        _, y_slice = Data.window.get_slice(session_tail, Data.sid)
        for i in range(1, len(y_slice)):
            y_o_idx = np.searchsorted(Data.classes, y_slice[i, 0])
            self.update_matrix(y_tail_idx, y_o_idx, -1)

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
            y_prev_idx = Data.session_vector[-1]
            co_counts = np.array(self.matrix.data[y_prev_idx])
            if len(co_counts) > 0:
                co_events = np.array(self.matrix.rows[y_prev_idx])
                y_proba[co_events] = co_counts / max(co_counts)
                y_proba[y_prev_idx] = 0.0
            predictions.append(y_proba)
        return np.array(predictions)

    def __str__(self):
        event_type = self.target_event_type or 'any'
        return f'{__class__.__name__} | event type: {event_type} | '
