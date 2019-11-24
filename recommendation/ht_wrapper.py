from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from collections import deque
from collections import defaultdict
import numpy as np
from collections import Counter
from skmultiflow.trees import HoeffdingTree
from utils.shared_data import SharedData as Data


class HTWrapper(BaseSKMObject, ClassifierMixin):
    """Wrapper for the Hoeffding Tree learner of Scikit-Multiflow

    Notes
    ----------
    Provides recommendation interface for the HoeffdingTree class
    Internally represented as a decision stump with Naive Bayes at the leaves

    References
    ----------
    "FlowRec: Prototyping Session-based Recommender Systems in Streaming Mode"

    Parameters
    ----------
    estimator: BaseSKMObject (default=HoeffdingTree)
        Estimator to wrap.
    weight_mc: float (default=10)
        Weight of the sequence 'previous item -> current item' (1st order Markov Chain)
    weight_inv: float (default=0.3)
        Weight of inverse sequences 'current item -> other item'
        Used as a fixed penalty for inverse sequences
    max_session_size: int (default=20)
        Cutoff for the session size. Used to filter out very long sessions.
    """

    def __init__(self,
                 estimator=HoeffdingTree(leaf_prediction='nb'),
                 weight_mc=10,
                 weight_inv=0.3,
                 max_session_size=20
                 ):
        super().__init__()
        self.ht = estimator
        self.w_mc = weight_mc
        self.w_inv = weight_inv
        self.counter = Counter()
        self.max_session_size = max_session_size
        self._rec_tracker = defaultdict(list)

    def configure(self, **kwargs):
        self.ht.classes = list(range(len(Data.classes)))
        self.ht.set_params(nominal_attributes=[0])
        self.ht.partial_fit(np.array([[-1]]), np.array([0]))

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            for i in range(row_cnt):
                y_idx = np.searchsorted(Data.classes, y)
                if Data.session_vector is not None:
                    session_vector = Data.session_vector[-self.max_session_size:]
                    for pos, y_o_idx in enumerate(session_vector):
                        if y_o_idx == session_vector[-1]:
                            w = self.w_mc
                        else:
                            w = 1 / (len(session_vector) - pos)
                        self.ht.partial_fit(np.array([[y_o_idx]]),
                                            y_idx, sample_weight=[w])
                        # inverse fit
                        self.ht.partial_fit(np.array([y_idx]),
                                            np.array([y_o_idx]),
                                            sample_weight=[w * self.w_inv])

    def predict(self, X):
        predictions = deque()
        r, _ = get_dimensions(X)
        y_proba = np.zeros((r, len(Data.classes)))
        for i in range(r):
            session_vector = Data.session_vector[-self.max_session_size:]
            for pos, y_o_idx in enumerate(session_vector):
                weight = self.w_mc if y_o_idx == session_vector[-1] else 1
                y_proba_current = self.ht.predict_proba(np.array([[y_o_idx]]))
                y_proba_current *= weight / (len(session_vector) - pos)
                y_proba += y_proba_current
            y_proba[i][Data.session_vector[-1]] = 0.0
            nonzero = np.flatnonzero(y_proba[i])
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
        """Not implemented for this method."""
        raise NotImplementedError
