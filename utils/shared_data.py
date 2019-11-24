class SharedData:
    """Holds useful shared data to be used by recommenders and some other classes.

    Notes
    -----
    Initialized by the evaluator (some of the variables are passed as arguments).
    None of the variables should be set directly in this file.
    """

    # =============================================================================
    # Variables set by the user during evaluator instantiation
    # =============================================================================

    # Column identifiers in input data
    sid = None  # Session column index (session_column_index parameter in EvaluatePrequential)
    tid = None  # Timestamp column index (time_column_index parameter in EvaluatePrequential)
    eid = None  # Event type column index (event_column_index parameter in EvaluatePrequential)

    # Whether to allow reminders and repeated recommendations
    allow_reminders = True  # allow_reminders parameter in EvaluatePrequential
    allow_repeated = True  # allow_repeated parameter in EvaluatePrequential

    # Recommendation size
    rec_size = 10  # rec_size parameter in EvaluatePrequential

    # Sliding window of observations
    window = None  # n_keep parameter in EvaluatePrequential

    # =============================================================================
    # Variables set automatically by the evaluator
    # =============================================================================

    # Array of all unique classes (labels) found in the input data
    classes = None

    # Vector of indexed items (in time order) seen so far in the current session
    session_vector = None
