# Example 2: event-specific rule-based recommenders
from skmultiflow.data import FileStream
from evaluation.evaluate_prequential import EvaluatePrequential
from recommendation.co_events import CoEventsClassifier
from recommendation.seq_events import SeqEventsClassifier

# Create stream
# Assume that the dataset contains event types at column 2 (0=click, 1=purchase)
stream = FileStream("your-dataset.csv")
stream.prepare_for_use()

# Instantiate recommenders
co_clicks = CoEventsClassifier(target_event_type=0)
click_click = SeqEventsClassifier(source_event_type=0, target_event_type=0)
click_purchase = SeqEventsClassifier(source_event_type=0, target_event_type=1)

# Configure evaluator
evaluator = EvaluatePrequential(session_column_index=0,
                                event_column_index=2,
                                rec_size=10,
                                pretrain_size=0,
                                n_wait=200,     # Evaluation window
                                n_keep=20000,   # Observation window
                                max_samples=100000,
                                show_plot=True,
                                metrics=['recall', 'mrr', 'running_time'])

# Run evaluation
evaluator.evaluate(stream=stream, 
                   model=[co_clicks, click_click, click_purchase], 
                   model_names=['co_clicks', 'click_click', 'click_purchase'])