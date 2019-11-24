from skmultiflow.data import FileStream
from evaluation.evaluate_prequential import EvaluatePrequential
from recommendation.random import RandomClassifier
from recommendation.oracle import OracleClassifier
from recommendation.popular import PopularClassifier
from recommendation.co_events import CoEventsClassifier
from recommendation.seq_events import SeqEventsClassifier
from recommendation.ht_wrapper import HTWrapper
from recommendation.beer import BeerEnsemble
from recommendation.sknn import SKNNClassifier

# 1. Create a stream
# attributes = '../data/yoochoose/attributes.csv'
# attributes = pd.read_csv("../data/yoochoose/attributes.csv")

stream = FileStream("../data/clef/clef_1M100K.csv")
# stream = FileStream("../data/yoochoose/events_only_bought_500k.csv") # 0: click, 1: purchase
# stream = FileStream("../data/yoochoose/clicks_1M100K.csv")
stream.prepare_for_use()

# 2. Instantiate recommenders
random = RandomClassifier()
oracle = OracleClassifier(stream)
ht = HTWrapper(weight_mc=5, weight_inv=0.01)
sknn = SKNNClassifier(k=200, sample_size=200, sample_recent=True, similarity='cosine')
popular = PopularClassifier(sliding_window=True)
ar = CoEventsClassifier(sliding_window=False)
sr = SeqEventsClassifier(sliding_window=False)
mc = SeqEventsClassifier(steps_back=1, sliding_window=False)

pop_clicks = PopularClassifier(event_type=0)
pop_purchases = PopularClassifier(event_type=1)
co_clicks = CoEventsClassifier(target_event_type=0)
co_purchases = CoEventsClassifier(target_event_type=1)

click_click = SeqEventsClassifier(source_event_type=0, target_event_type=0)
# click_cart = SeqEventsClassifier(source_event_type=1, target_event_type=2)
# cart_purchase = SeqEventsClassifier(source_event_type=2, target_event_type=3)
click_purchase = SeqEventsClassifier(source_event_type=0, target_event_type=1)
purchase_click = SeqEventsClassifier(source_event_type=1, target_event_type=0)

beer = BeerEnsemble(cf_components=[ar, sr, mc, popular, sknn],
                    boundaries=[],
                    verbose=False)

# 3. Setup the evaluator
evaluator = EvaluatePrequential(session_column_index=0,
                                time_column_index=1,
                                rec_size=10, #D
                                allow_reminders=True,
                                allow_repeated=True,
                                show_plot=False,
                                n_wait=10000, # evaluation window size
                                n_keep=10000,  # training window size
                                n_skip=100000,
                                pretrain_size=10,
                                max_samples=10000,
                                metrics=['recall', 'mrr', 'running_time'])

# 4. Run evaluation
# evaluator.evaluate(stream=stream, model=[ht, hat, hatt], model_names=['HT', 'HAT', 'HATT'])
# evaluator.evaluate(stream=stream, model=[ht], model_names=['hoeffding tree'])
# evaluator.evaluate(stream=stream, model=[hte], model_names=['ht ensemble'])
evaluator.evaluate(stream=stream, model=[ht], model_names=['ht'])
# evaluator.evaluate(stream=stream, model=[sknn], model_names=['sknn'])
# evaluator.evaluate(stream=stream, model=[beer], model_names=['BEER[TS]'])
# evaluator.evaluate(stream=stream, model=[beer_simple], model_names=['BEER[TS] simple'])
# evaluator.evaluate(stream=stream, model=[ar], model_names=['ar'])
# evaluator.evaluate(stream=stream, model=[random, oracle], model_names=['random', 'oracle'])
# evaluator.evaluate(stream=stream, model=[ar, sr, mc, beer, beer_simple],
#                    model_names=['ar', 'sr', 'mc', 'beer', 'beer_old'])
# evaluator.evaluate(stream=stream, model=[popular], model_names=['popular'])
# evaluator.evaluate(stream=stream, model=[random], model_names=['random'])
# evaluator.evaluate(stream=stream, model=[oracle], model_names=['oracle'])
# evaluator.evaluate(stream=stream, model=[click_click], model_names=['click_click'])
# evaluator.evaluate(stream=stream, model=[ar, sr, mc, popular, random, sknn, beer, ht],
#                     model_names=['AR', 'SR', 'MC', 'POP', 'RAND', 'S-KNN', 'BEER[TS]', 'HT'])
# evaluator.evaluate(stream=stream, model=[pop_clicks, click_click, co_clicks],
#                     model_names=['pop_clicks', 'click_click', 'co_clicks'])
