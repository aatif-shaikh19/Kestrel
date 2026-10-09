import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

POST_W = 3.0   # post-1-May-2026 rows count 3x: the approval regime changed, old rows are less representative

def weights(ts):
    return np.where(pd.to_datetime(ts) >= pd.Timestamp("2026-05-01"), POST_W, 1.0)

def fit_models(X, y, ts, n_seeds=5):
    sw = weights(ts)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=4000))
    lr.fit(X.fillna(0), y, logisticregression__sample_weight=sw)
    gbs = [HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                          l2_regularization=3.0, random_state=s).fit(X, y, sample_weight=sw) for s in range(n_seeds)]
    return {"lr": lr, "gbs": gbs, "cols": list(X.columns)}

def predict(models, X):
    X = X[models["cols"]]
    p_lr = models["lr"].predict_proba(X.fillna(0))[:, 1]
    p_gb = np.mean([m.predict_proba(X)[:, 1] for m in models["gbs"]], axis=0)
    return 0.5 * p_lr + 0.5 * p_gb, p_lr, p_gb
