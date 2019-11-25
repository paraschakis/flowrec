import sys
sys.path.append("..")
from skmultiflow.data import FileStream
from evaluation.evaluate_prequential import EvaluatePrequential
from recommendation.random import RandomClassifier
from recommendation.popular import PopularClassifier
from recommendation.co_events import CoEventsClassifier
from recommendation.seq_events import SeqEventsClassifier
from recommendation.ht_wrapper import HTWrapper
from recommendation.beer import BeerEnsemble
from recommendation.sknn import SKNNClassifier

# Create stream
stream = FileStream("../data/trivago_1M100K.csv")
stream.prepare_for_use()

# Instantiate recommenders
random = RandomClassifier()
ht = HTWrapper(weight_mc=5, weight_inv=0.90)
sknn = SKNNClassifier(k=100, sample_size=1000, sample_recent=True,
                      similarity='cosine', sliding_window=True)
popular = PopularClassifier(sliding_window=True)
ar = CoEventsClassifier(sliding_window=False)
sr = SeqEventsClassifier(sliding_window=False)
mc = SeqEventsClassifier(steps_back=1, sliding_window=False)
beer = BeerEnsemble(cf_components=[ar, sr, mc, popular, sknn])

evaluator = EvaluatePrequential(session_column_index=0,
                                time_column_index=1,
                                rec_size=10,
                                allow_reminders=True,
                                allow_repeated=True,
                                show_plot=True,
                                n_wait=10000,
                                n_keep=50000,
                                n_skip=100000,
                                pretrain_size=0,
                                max_samples=1000000,
                                metrics=['recall', 'mrr', 'running_time'])

# Run prequential evaluation
evaluator.evaluate(stream=stream, model=[ar, sr, mc, popular, random, sknn, beer, ht],
                   model_names=['AR', 'SR', 'MC', 'POP', 'RAND', 'S-KNN', 'BEER[TS]', 'HT'])
