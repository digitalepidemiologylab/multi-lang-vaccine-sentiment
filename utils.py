import numpy as np
import sklearn.metrics


def performance_metrics(y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
    """
    Compute performance metrics

    By default computes accuracy, precision, recall and f1 for 4 type of averagings (micro, macro, weighted and per class) 

    Parameters:
    y_true (list): True values
    y_pred (list): Predicted values
    metrics (list): By default ['accuracy', 'precision', 'recall', 'f1']
    averaging (array): By default ['micro', 'macro', 'weighted', None]
    label_mapping (dict): Maps integers to label names

    Returns:
    scores (dict): Performance metrics

    -------------------------------------------------------
    Example usage:

    y_true  = [0,1,1,2,1]
    y_pred  = [0,1,2,2,1]
    label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
    scores = performance_metrics(y_true, y_pred, label_mapping=label_mapping)
    print(scores)
    {
        'accuracy': 0.8,
         'f1_macro': 0.8222222222222223,
         'f1_micro': 0.8000000000000002,
         'f1_negative': 0.8,
         'f1_neutral': 0.6666666666666666,
         'f1_positive': 1.0,
         'f1_weighted': 0.8133333333333335,
         'precision_macro': 0.8333333333333334,
         'precision_micro': 0.8,
         'precision_negative': 1.0,
         'precision_neutral': 0.5,
         'precision_positive': 1.0,
         'precision_weighted': 0.9,
         'recall_macro': 0.8888888888888888,
         'recall_micro': 0.8,
         'recall_negative': 0.6666666666666666,
         'recall_neutral': 1.0,
         'recall_positive': 1.0,
         'recall_weighted': 0.8
     }
    """

    def _compute_performance_metric(scoring_function, m, y_true, y_pred):
        for av in averaging:
            if av is None:
                metrics_by_class = scoring_function(y_true, y_pred, average=av, labels=labels)
                for i, class_metric in enumerate(metrics_by_class):
                    if label_mapping is None:
                        label_name = labels[i]
                    else:
                        label_name = label_mapping[labels[i]]
                    scores[m + '_' + str(label_name)] = class_metric
            else:
                scores[m + '_' + av] = scoring_function(y_true, y_pred, average=av, labels=labels)
    if averaging is None:
        averaging = ['micro', 'macro', 'weighted', None]
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {}
    labels = sorted(np.unique(y_true))
    # label_mapping = {v: k for k, v in label_mapping.items()}


    if len(labels) <= 2:
        # binary classification
        averaging += ['binary']
    for m in metrics:
        if m == 'accuracy':
            scores[m] = sklearn.metrics.accuracy_score(y_true, y_pred)
        elif m == 'precision':
            _compute_performance_metric(sklearn.metrics.precision_score, m, y_true, y_pred)
        elif m == 'recall':
            _compute_performance_metric(sklearn.metrics.recall_score, m, y_true, y_pred)
        elif m == 'f1':
            _compute_performance_metric(sklearn.metrics.f1_score, m, y_true, y_pred)
    return scores
