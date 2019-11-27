<img src="https://github.com/flowrec/flowrec/blob/master/FlowRec.png">

<!---# FlowRec
**Prototyping Session-based Recommender Systems in Streaming Mode.**-->

 - Implemented on top of [Scikit-Multiflow](https://scikit-multiflow.github.io/).
 - Uses prequential evaluation for online learning.
 - Contains baseline recommendation algorithms.
 
## Installation
 1. Install [Anaconda distribution](https://www.anaconda.com/distribution/) with Python >= 3.5.
 2. Install [Scikit-Multiflow](https://scikit-multiflow.github.io/scikit-multiflow/installation.html).
 3. Clone [FlowRec repository](https://github.com/flowrec/flowrec.git).
 
## Example
```Python
from skmultiflow.data import FileStream
from evaluation.evaluate_prequential import EvaluatePrequential
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
                                n_wait=200,      # evaluation window
                                n_keep=20000,    # observation window
                                max_samples=100000,
                                metrics=['recall', 'mrr', 'running_time'])

# Run evaluation
evaluator.evaluate(stream=stream, model=[popular], model_names=['POP'])

```
 
Stay tuned for more examples and tutorials (coming soon...)
