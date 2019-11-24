import numpy as np
import pandas as pd
from skmultiflow.core import BaseSKMObject, MetaEstimatorMixin
from skmultiflow.utils import get_dimensions
from recommendation.attribute import AttributeClassifier
from collections import Counter
from collections import OrderedDict
from collections import deque
from collections import namedtuple
from collections import defaultdict
import random
from utils.shared_data import SharedData as Data


class BeerEnsemble(BaseSKMObject, MetaEstimatorMixin):
    """Bandit-based ensemble of component recommenders based on Thompson Sampling

    Terminology
    ----------
    - a predictor is a component that has been split into groups
    - a group is a part of a component that corresponds to a specific precision range

    Notes
    ----------
    Any recommender implementing the predict_proba() method can be used as a component.
    Component partitioning (optional) is achieved by providing the boundaries for prediction probabilities.

    Optionally accepts a file with item-to-attributes mappings for building attribute-based components
    Expected format: first column with item ids, and other columns with attribute values

    References
    ----------
    Brodén et al. (2019). In ACM Trans. Interact. Intell. Syst. 10, 1, Article 4.
    "A Bandit-Based Ensemble Framework for Exploration/Exploitation of Diverse Recommendation Components:
    An Experimental Study within E-Commerce"
    """

    def __init__(self, cf_components=[], attr_file=None, boundaries=[], verbose=False):
        super().__init__()
        self.cf_components = cf_components
        self.attr_file = attr_file
        self.verbose = verbose
        self.components = None
        self.sampler = None
        self.current_predictions = OrderedDict()  # dict that holds who recommended what
        self.current_sleeping = set()
        self.query_counter = Counter()
        self.query_miss_counter = Counter()
        self.responses = dict()
        self._rec_tracker = defaultdict(set)
        if any(b for b in boundaries if b < 0 or b > 1):
            raise ValueError('The boundaries of prediction probabilities should be from 0 to 1')
        else:
            self.boundaries = boundaries
        if len(self.cf_components) == 0 and not attr_file:
            raise ValueError('The ensemble is empty. Please provide at least one component.')

    def configure(self, **kwargs):
        self.components = self.cf_components + self._build_attr_components()
        for c in self.components:
            if hasattr(c, 'configure'):
                c.configure(**kwargs)
        self.sampler = self.Sampler(self.components, self.boundaries)

    def _build_attr_components(self):
        attr_components = []
        if self.attr_file is not None:
            try:
                attr_df = pd.read_csv(self.attr_file)
                # retain only observed items
                attr_df = attr_df[attr_df.iloc[:, 0].isin(Data.classes)]
                attr_components = [AttributeClassifier(attr_df.iloc[:, [0, i]])
                                   for i in range(1, len(attr_df.columns))]
            except FileNotFoundError:
                print('Attribute file not found. Only behavioral components will be used.')
        return attr_components

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: numpy.ndarray (default=None)
            Array with all possible/known class labels.

        sample_weight: Not used (default=None)

        Returns
        -------
        AdditiveExpertEnsemble
            self
        """

        for i in range(len(X)):
            self._fit_single_sample(X[i:i + 1, :], y[i:i + 1], classes, sample_weight)
        return self

    def _fit_single_sample(self, X, y, classes=None, sample_weight=None):
        for component in self.components:
            component.partial_fit(X, y, classes, sample_weight)
        if len(self.current_predictions) > 0:
            self.sampler.update_posterior(self.current_predictions, y[0])

    def predict(self, X):
        """

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """

        predictions = deque()
        r, _ = get_dimensions(X)
        for i in range(r):
            self.responses.clear()
            self.current_predictions.clear()
            self.current_sleeping.clear()
            for rank in range(Data.rec_size):
                self._predict_single_item(X[i:i + 1, :])
            y_pred = np.fromiter(self.current_predictions.keys(), dtype=int)
        predictions.append(y_pred)
        return np.array(predictions)

    def _predict_single_item(self, X):
        sorted_samples = self.sampler.sample()
        for predictor in sorted_samples:
            if predictor not in self.current_sleeping:
                self.query_counter[predictor] += 1
                item = self._get_response(X, predictor)
                if item is not None:
                    self.current_predictions[item] = predictor
                    return
                else:
                    self.current_sleeping.add(predictor)
                    self.query_miss_counter[predictor] += 1

    def _get_response(self, X, predictor):
        try:
            y_proba = self.responses[predictor.id_]
        except KeyError:
            y_proba = self.components[predictor.id_].predict_proba(X)
            y_proba[0][Data.session_vector[-1]] = 0.0
            self.responses[predictor.id_] = y_proba
        num_predicted = np.count_nonzero(y_proba[0])
        if num_predicted > 0:
            interval = self._get_interval(predictor)
            sorted_ids = y_proba[0].argsort()[::-1][:num_predicted]
            session = X[0, Data.sid]
            for i in sorted_ids:
                item = Data.classes[i]
                # if best prediction is below interval, no point to continue
                if y_proba[0][i] < interval[0]:
                    return None
                elif y_proba[0][i] <= interval[1] and \
                        item not in self.current_predictions:
                    if not Data.allow_reminders and i in Data.session_vector:
                        continue
                    if not Data.allow_repeated and i in self._rec_tracker[session]:
                        continue
                    self._rec_tracker[session].add(i)
                    return item

    def _get_interval(self, predictor):
        if len(self.boundaries) == 0 or predictor.id_ >= len(self.cf_components):
            return 0, 1  # TODO: IDF-based split for attribute-based components
        else:
            if predictor.group == 0:
                return 0, self.boundaries[0]
            elif predictor.group == len(self.boundaries):
                return self.boundaries[-1], float('inf')
            else:
                return self.boundaries[predictor.group - 1], self.boundaries[predictor.group]

    def predict_proba(self, X):
        """ Not implemented for this method. """
        raise NotImplementedError

    def display_info(self):
        if self.verbose:
            print('\nBEER ensemble statistics\n')
            total_displays = sum([sum(beta_params) - 2
                                  for beta_params in self.sampler.predictors.values()])
            for predictor, beta_params in self.sampler.predictors.items():
                print(f'{self.components[predictor.id_]}({predictor.group})')
                successes, failures = beta_params[0] - 1, beta_params[1] - 1
                displays = successes + failures
                coverage = 1 - self.query_miss_counter[predictor] / self.query_counter[predictor]
                output = (
                    f'successes/failures/displays: {successes}/{failures}/{displays}\n'
                    f'success ratio: {successes / displays:.3f}\n'
                    f'exposure ratio: {displays / total_displays:.3f}\n'
                    f'query coverage: {coverage:.3f}\n'
                    f'-------------'
                )
                print(output)

    class Sampler:

        def __init__(self, components, boundaries=[]):
            self.predictors = OrderedDict()
            Predictor = namedtuple('Predictor', ['id_', 'group'])
            for i, component in enumerate(components):
                if component.__class__.__name__ == 'AttributeClassifier':
                    # TODO: split attribute-based components
                    self.predictors[Predictor(id_=i, group=0)] = np.ones(2, dtype=int)
                else:
                    for j in range(len(boundaries) + 1):
                        self.predictors[Predictor(id_=i, group=j)] = np.ones(2, dtype=int)

        def sample(self):
            """Thompson Sampling."""
            sorted_predictors = sorted(self.predictors.keys(), key=lambda k:
                                       self.beta_sample(k), reverse=True)
            return sorted_predictors

        def beta_sample(self, group_idx):
            a = self.predictors[group_idx][0]  # Success
            b = self.predictors[group_idx][1]  # Failure
            return random.betavariate(a, b)

        def update_posterior(self, current_predictions, y_true):
            for y_pred, group_idx in current_predictions.items():
                if y_pred == y_true:
                    self.predictors[group_idx][0] += 1
                else:
                    self.predictors[group_idx][1] += 1
