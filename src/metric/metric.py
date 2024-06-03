import sklearn
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(
    y_pred,
    y_true,
):
    return cohen_kappa_score(
        y_true.astype(int),
        y_pred.clip(0, 5).round(0),
        weights='quadratic',
    )