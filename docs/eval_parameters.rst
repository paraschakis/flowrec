==================
FlowRec parameters
==================

This is a list of FlowRec-specific parameters for prequential evaluation in addition to those originally provided by Scikit-Multiflow (see `here <https://scikit-multiflow.github.io/scikit-multiflow/_autosummary/skmultiflow.evaluation.EvaluatePrequential.html#skmultiflow.evaluation.EvaluatePrequential>`_).

:session_column_index (required):
    The index of a data column containing session identifiers.
:time_column_index (optional):
    The index of a data column containing timestamps. Note that they are not used by the stream to order the samples.
:event_column_index (optional):
    The index of a data column containing indexed event types. This allows for event-specific training and evaluation.
:rec_triggers (optional):
    List of indexed event types for which recommendations are requested (only used when event_column_index is provided).
:allow_repeated (default=True):
    Allow/disallow previously suggested items to be recommended again in subsequent events of a session.
:allow_reminders (default=True):
    Allow/disallow previously visited items to be recommended in subsequent events of a session, thus acting as reminders.
:rec_size (default=10):
    The size of the recommendation list (a.k.a. cutoff).
:n_skip (default=0):
    The number of samples to skip from the start of the stream.
:n_keep (default=1000):
    The size of the observation window.