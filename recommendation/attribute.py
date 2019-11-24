from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
import numpy as np
from collections import Counter
from collections import defaultdict
from utils.shared_data import SharedData as Data
import pandas as pd

pd.set_option('mode.chained_assignment', None)


class AttributeClassifier(BaseSKMObject, ClassifierMixin):
    """Classifier based on single attributes.

    Notes
    ----------
    The prediction scores are ranked by popularity

    Parameters
    ----------
    attr_data: Pandas dataframe
        Dataframe containing item-attribute mappings
    """

    def __init__(self, attr_data):
        super().__init__()
        self.counter = Counter()  # popularity counter
        attr_data.iloc[:, 0] = attr_data.iloc[:, 0].apply(self._get_idx)
        self.attr_data = attr_data
        self._rec_tracker = defaultdict(list)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            for i in range(row_cnt):
                self.counter[self._get_idx(y[i])] += 1  # total popularity
        return self

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
            value = self.attr_data[self.attr_data.iloc[:, 0] == y_prev_idx].iat[0, 1]
            y_pred_df = self.attr_data[self.attr_data.iloc[:, 1] == value]
            y_pred_idx_df = y_pred_df.iloc[:, 0]
            max_count = max(self.counter.values())
            y_pred_proba = np.fromiter(((self.counter[y] + 1.0) / max_count
                                        for y in y_pred_idx_df), dtype=float)
            y_proba[y_pred_idx_df] = y_pred_proba
            predictions.append(y_proba)
        return np.array(predictions)

    @staticmethod
    def _get_idx(y):
        return np.where(Data.classes == y)[0][0]

    def __str__(self):
        return f'{__class__.__name__} ({self.attr_data.columns[1]})'
