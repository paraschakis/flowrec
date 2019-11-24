from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import get_dimensions
from collections import deque
import numpy as np
from collections import Counter
from collections import defaultdict
from utils.shared_data import SharedData as Data


class SKNNClassifier(BaseSKMObject, ClassifierMixin):
    """Session-based K-Nearest Neighbors recommender

    Notes
    ----------
    Uses a sub-sampling trick to speed up the computation, as described in the reference.

    References
    ----------
    Jannach & Ludewig (2017). In 11th ACM Conference on Recommender Systems
    "When Recurrent Neural Networks Meet The Neighborhood For Session-based Recommendation"

    Parameters
    ----------
    k: int (default=100)
        Size of the neighborhood.
    similarity: string (default='cosine')
        Similarity measure.
    sample_size: int (default=0)
        How many potential neighbors to sub-sample. Use 0 for all neighbors.
    sample_recent: boolean (default=True)
        Whether to sub-sample recent or random potential neighbors.
    sliding_window: boolean (default=True)
        Whether to look for neighbors only within the sliding window
    """

    def __init__(self, k=100, sample_size=0, sample_recent=True,
                 sliding_window=True, similarity='cosine'):
        super().__init__()
        self.k = k
        self.similarity = similarity
        self.sample_size = sample_size  # sample size of 0 means all sessions
        self.sample_recent = sample_recent
        self.sliding_window = sliding_window
        self._rec_tracker = defaultdict(list)
        self.session_items = defaultdict(set)
        self.item_sessions = defaultdict(set)
        self.ordered_sessions = np.array([], dtype=int)
        self._num_examples = 0

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        r, _ = get_dimensions(X)
        for i in range(r):
            # Delete oldest session-item pair
            if self.sliding_window and self._num_examples > Data.window.max_size:
                x_tail = Data.window.get_attributes_matrix()[0]
                y_tail = Data.window.get_targets_matrix()[0]
                session_tail = x_tail[Data.sid]
                item_tail = np.searchsorted(Data.classes, y_tail)[0]
                _, y_slice = Data.window.get_slice(session_tail, Data.sid)
                occurrences = np.nonzero(y_slice[:, 0] == y_tail[0])[0]
                if len(occurrences) == 1:  # only delete if no duplicates exist
                    try:
                        self.session_items[session_tail].remove(item_tail)
                        self.item_sessions[item_tail].remove(session_tail)
                    except KeyError:
                        pass
            # Add session-item pair
            session = X[i, Data.sid]
            item = np.searchsorted(Data.classes, y[0])
            self.session_items[session].add(item)
            self.item_sessions[item].add(session)
            self.ordered_sessions = self.ordered_sessions[self.ordered_sessions != session]
            self.ordered_sessions = np.append(self.ordered_sessions, session)
            self._num_examples += 1

    def predict(self, X):
        predictions = deque()
        r, _ = get_dimensions(X)
        y_proba = self.predict_proba(X)
        for i in range(r):
            nonzero = np.nonzero(y_proba[i])[0]
            if len(nonzero > 0):
                sorted_desc = np.argsort(y_proba[i][nonzero])[::-1]
                sorted_ids = nonzero[sorted_desc]
                if not Data.allow_reminders:
                    sorted_ids = sorted_ids[~np.isin(sorted_ids, Data.session_vector)]
                if not Data.allow_repeated:
                    session = X[i, Data.sid]
                    sorted_ids = sorted_ids[~np.isin(sorted_ids,
                                                     self._rec_tracker[session])]
                    self._rec_tracker[session].extend(sorted_ids[:Data.rec_size])
                y_pred = Data.classes[sorted_ids[:Data.rec_size]]
            else:
                y_pred = np.array([])
            predictions.append(y_pred)
        return np.array(predictions)

    def predict_proba(self, X):
        predictions = deque()
        r, _ = get_dimensions(X)
        for i in range(r):
            y_proba = np.zeros(len(Data.classes))
            scored_neighbors = self._find_neighbors()
            for neighbor in scored_neighbors:
                neighbor_items = list(self.session_items[neighbor[0]])
                y_proba[neighbor_items] += neighbor[1]
            nonzero = np.nonzero(y_proba)[0]
            if len(nonzero) > 0:
                y_proba[nonzero] /= max(y_proba[nonzero])
                y_proba[Data.session_vector[-1]] = 0.0
            predictions.append(y_proba)
        return np.array(predictions)

    def _find_neighbors(self):
        neighbors = set()
        if self.sample_size == 0:  # Consider all sessions
            neighbors = set(self.session_items.keys())
        else:
            for item in Data.session_vector:
                neighbors |= self.item_sessions[item]
            if len(neighbors) > self.sample_size:
                if self.sample_recent:
                    neighbor_ids = np.nonzero(np.isin(self.ordered_sessions, list(neighbors)))[0]
                    neighbors = self.ordered_sessions[neighbor_ids][-self.sample_size:]
                else:
                    neighbors = list(np.random.choice(list(neighbors),
                                                      self.sample_size,
                                                      replace=False))

        nearest_neighbors = self._get_nearest(neighbors)
        return nearest_neighbors

    def _get_nearest(self, neighbors):
        neighbors_scores = Counter()
        if len(neighbors) > 0:
            set_current = set(Data.session_vector)
            for neighbor in neighbors:
                set_neighbor = self.session_items[neighbor]
                neighbors_scores[neighbor] = self._calc_score(set_current, set_neighbor)
        return neighbors_scores.most_common(self.k)

    def _calc_score(self, set1, set2):
        if set1 == set2:
            return 1
        intersection_size = len(set1 & set2)
        sum_cardinalities = len(set1) + len(set2)
        if self.similarity == 'jaccard':
            return intersection_size / len(set1 | set2)
        elif self.similarity == 'dice':
            return 2 * intersection_size / sum_cardinalities
        elif self.similarity == 'tanimoto':
            return intersection_size / (sum_cardinalities - intersection_size)
        else:  # use cosine
            return intersection_size / (np.sqrt(len(set1) * len(set2)))
