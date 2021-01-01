import numpy as np
from sklearn import metrics
from scipy import sparse
from collections.abc import Iterable


def y_to_sparse(y):
    y_len = len(y)
    return sparse.csr_matrix((np.ones(y_len, dtype=int), (list(range(y_len)), y)))


def y_hat_to_sparse(y_pred):
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)

    y_pred = y_pred.astype(int)
    y_pred_len = len(y_pred)

    return sparse.csr_matrix((np.ones(y_pred_len, dtype=int), (list(range(y_pred_len)), y_pred)))


def compute_classification_metrics(y_train, y_preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_preds)
    return {
        'acc': metrics.accuracy_score(y_train, y_preds),
        'auc': metrics.auc(fpr, tpr),
        'precision': metrics.precision_score(y_train, y_preds),
        'recall': metrics.recall_score(y_train, y_preds),
        'f1': metrics.f1_score(y_train, y_preds),
        'filtering': sum(y_preds) / len(y_preds)
    }


def format_compound_value(value):
    if not isinstance(value, str) and isinstance(value, Iterable):
        ','.join([str(item) for item in value])
    else:
        return value


def format_nested_parameters(param_dict, param_name):
    return {f'{param_name}__{key}': format_compound_value(value)
            for key, value in param_dict.items()}
