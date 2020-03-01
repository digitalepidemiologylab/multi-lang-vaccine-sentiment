import numpy as np
import sklearn.metrics

#Define some custom functions
def performance_metrics(y_true, y_pred, metrics=None, averaging=None, label_mapping=None):
    """
    Compute performance metrics
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
    if label_mapping is None:
        # infer labels from data
        labels = sorted(list(set(y_true + y_pred)))
    else:
        labels = sorted(list(label_mapping.keys()))
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

def get_full_output(probabilities, y_true, label_mapping=None):
    results = []
    probabilities = np.array(probabilities)
    assert len(probabilities) == len(y_true)
    for i in range(len(probabilities)):
        sorted_ids = np.argsort(-probabilities[i])
        if label_mapping is None:
            labels = sorted_ids
        else:
            labels = [label_mapping[s] for s in sorted_ids]
        results.append({
            'prediction': labels[0],
            'predictions': labels,
            'y_true': y_true[i],
            'probability': probabilities[i][sorted_ids][0],
            'probabilities': probabilities[i][sorted_ids]
            })
    return results
