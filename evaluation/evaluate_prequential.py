import os
import warnings
import re
from timeit import default_timer as timer
from collections import Counter
import numpy as np
from evaluation.base_evaluator import StreamEvaluator
from utils import constants
from utils.data_structures import InstanceWindow
from utils.shared_data import SharedData as Data


class EvaluatePrequential(StreamEvaluator):
    """ The prequential evaluation method or interleaved test-then-train method.

    The prequential evaluation is designed specifically for stream settings,
    in the sense that each sample serves two purposes, and that samples are
    analysed sequentially, in order of arrival.

    This method consists of using each sample to test the model, which means
    to make predictions, and then the same sample is used to train the model
    (partial fit). This way the model is always tested on samples that it
    hasn't seen yet.

    This evaluator can process a single learner to track its performance
    or multiple learners at a time, to compare different models on the same stream.

    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test (e.g. evaluation window size).
        Also defines when to update the plot if `show_plot=True`.
        Note that setting `n_wait` too small can significantly slow down the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    n_skip: int (Default: 0)
        The number of samples to skip before the evaluation.

    n_keep: int (Default: 1000)
        The number of samples to keep in the observation history window.

    allow_repeated: boolean (Default: True)
        Whether to allow repeated recommendations (in the same session)
        Repeated means that the same item was recommended earlier in the session.

    allow_reminders: boolean (Default: True)
        Whether to allow reminders (in the same session)
        Reminders are recommendations of items that were visited earlier in the session.

    session_column_index: int
        The index of the column in the input file that contains session identifiers.

    time_column_index: int (Default=None)
        The index of the column in the input file that contains timestamps.

    event_column_index: int (Default=None)
        The index of the column in the input file that contains event types (numeric only).

    rec_triggers: list (Default=None)
        List of event types (numeric) that trigger recommendation requests.
        E.g. produce recommendations for click events, but not for add-to-cart events.

    rec_size: list (Default=10)
        The size of the recommendation list.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).
        Currently not used.

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the evaluation.
        Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['recall', 'mrr'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | 'precision'
        | 'recall'
        | 'mrr'
        | 'F1'
        | 'running_time'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    data_points_for_classification: bool(Default: False)
        If True , the visualization used is a cloud of data points
    """

    def __init__(self,
                 session_column_index,
                 time_column_index=None,
                 event_column_index=None,
                 rec_triggers=None,
                 rec_size=10,
                 allow_repeated=False,
                 allow_reminders=False,
                 n_wait=200,
                 n_keep=1000,
                 n_skip=0,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=0,
                 max_time=float("inf"),
                 metrics=['recall', 'mrr'],
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False):

        super().__init__()
        self._method = 'prequential'
        self.rec_size = rec_size  # D
        self.allow_repeated = allow_repeated
        self.allow_reminders = allow_reminders
        self.rec_triggers = rec_triggers
        self.n_wait = n_wait
        self.n_keep = n_keep
        self.n_skip = n_skip
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.sid = session_column_index
        self.tid = time_column_index
        self.eid = event_column_index
        self.observation_window = InstanceWindow(max_size=self.n_keep)

        if metrics is None and data_points_for_classification is False:
            self.metrics = [constants.ACCURACY]

        elif data_points_for_classification is True:
            self.metrics = [constants.DATA_POINTS]

        else:
            self.metrics = metrics

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """

        # Populate shared data
        Data.sid = self.sid
        Data.tid = self.tid
        Data.eid = self.eid
        Data.window = self.observation_window
        Data.classes = np.unique(stream.target_values)
        Data.class_ids = list(range(len(Data.classes)))
        Data.allow_reminders = self.allow_reminders
        Data.allow_repeated = self.allow_repeated
        Data.rec_size = self.rec_size

        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()

            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        session_counter = Counter()

        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        if self.tid is None:
            test_cols = [self.sid]
        else:
            test_cols = [self.sid, self.tid]

        if self.n_skip > 0:
            self.stream.next_sample(self.n_skip)

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)
            session_counter.update(X[:, self.sid])

            # Pre-training
            for j in range(self.pretrain_size):
                for i in range(self.n_models):
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X[j:j + 1], y=y[j:j + 1])
                    self.running_time_measurements[i].compute_training_time_end()
                    self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
                self.observation_window.add_element(X[j:j + 1], y[j:j + 1])
            self.global_sample_count += self.pretrain_size

        update_count = 0
        evaluation_count = 0

        # Start evaluation.
        print('Evaluating...')
        while ((self.global_sample_count < actual_max_samples) &
               (self._end_time - self._start_time < self.max_time) &
               (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)  # Batches are currently not supported
                prediction = [None] * self.n_models
                try:  # In case of no pre-training, the observation window is empty
                    window_sessions = Data.window.get_attributes_matrix()[:, self.sid]
                except IndexError:
                    window_sessions = np.array([])
                if self.eid is not None:
                    event_type = X[0, [self.eid]][0]
                session = X[0, [self.sid]][0]
                session_counter[session] += 1
                Data.session_vector = self._get_indexed_session_vector(session)
                inputs_exist = X is not None and y is not None
                is_rec_trigger = (self.rec_triggers is None or
                                  self.eid is None or
                                  event_type in self.rec_triggers)
                is_known_session = session in window_sessions
                if inputs_exist and is_rec_trigger and is_known_session:
                    # Evaluate only on known sessions and on events that are recommendation triggers
                    evaluation_count += 1
                    for i in range(self.n_models):
                        try:
                            self.running_time_measurements[i].compute_testing_time_begin()
                            X_test = np.full(X.shape, None)
                            X_test[:, test_cols] = X[:, test_cols]  # Set all but test columns to None

                            # Generate and index recommendations
                            prediction[i] = self.model[i].predict(X_test)[0]
                            if len(prediction[i]) == 0:
                                pred_ids = np.array([-1])
                            else:
                                pred_ids = np.searchsorted(Data.classes, prediction[i])[:self.rec_size]

                            # Get index of true label
                            y_idx = np.searchsorted(Data.classes, y[0])
                            # Calculate metrics
                            self.mean_eval_measurements[i].add_result(y_idx, pred_ids)
                            self.current_eval_measurements[i].add_result(y_idx, pred_ids)
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))

                if y is not None:
                    # Train
                    for i in range(self.n_models):
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(X, y)
                        self.running_time_measurements[i].compute_training_time_end()
                        self.running_time_measurements[i].update_time_measurements(self.batch_size)

                self.global_sample_count += self.batch_size
                self._check_progress(actual_max_samples)
                self.observation_window.add_element(X, y)  # Add event to the sliding window

                if ((self.global_sample_count % self.n_wait) == 0 or
                        (self.global_sample_count >= self.max_samples) or
                        (self.global_sample_count / self.n_wait > update_count + 1) or
                        (evaluation_count == 1)):
                    self._update_metrics()
                    update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        self.evaluation_summary()

        try:
            self.model[0].display_info()
        except AttributeError:
            pass

        print('training time window: {}'.format(self.n_keep))
        print('number of sessions: {}'.format(len(session_counter)))
        print('number of evaluations: {}'.format(evaluation_count))
        print('avg. session size: {0:.2f}'.format(np.mean(list(session_counter.values()))))
        # evaluated_sessions_sizes = [c for c in session_counter.values() if c != 1]
        # print('average session size: {0:.2f}'.format(np.mean(evaluated_sessions_sizes)))
        
        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the recommendation labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the recommendation task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """

        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))
        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info

    @staticmethod
    def _get_indexed_session_vector(session):
        """Builds session representation containing indices of consumed items."""
        _, y_slice = Data.window.get_slice(session, Data.sid)
        try:
            return np.searchsorted(Data.classes, y_slice[:, 0])
        except IndexError:
            return np.array([])
