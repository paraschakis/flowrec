# Example 1: popularity recommender
from skmultiflow.data import FileStream
from evaluation.evaluate_prequential import EvaluatePrequential
from recommendation.random import RandomClassifier
from recommendation.popular import PopularClassifier

# Create stream
stream = FileStream("your-dataset.csv")
stream.prepare_for_use()

# Instantiate recommender
popular = PopularClassifier(sliding_window=True)

# Configure evaluator
evaluator = EvaluatePrequential(session_column_index=0,
                                rec_size=10,
                                pretrain_size=0,
                                n_wait=200,     # evaluation window
                                n_keep=20000,   # observation window
                                max_samples=100000,
                                metrics=['recall', 'mrr', 'running_time'])

# Run evaluation
evaluator.evaluate(stream=stream, model=[popular], model_names=['POP'])
