from sklearn.metrics import roc_auc_score, average_precision_score


def aps(results, labels):
    return average_precision_score(y_true=labels, y_score=results)


def auc(results, labels):
    return roc_auc_score(y_true=labels, y_score=results)



