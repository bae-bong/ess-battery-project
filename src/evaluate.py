"""평가 함수. 모델은 log10(수명)을 예측하므로, 지표는 원래 단위(사이클)로 되돌려서 계산한다."""
import numpy as np
import pandas as pd
from sklearn.base import clone


def to_cycles(y_log):
    return 10 ** np.asarray(y_log, dtype=float)


def metrics(y_log_true, y_log_pred):
    t, p = to_cycles(y_log_true), to_cycles(y_log_pred)
    return {
        'MAPE': float(np.mean(np.abs(t - p) / t) * 100),   # 평균 절대 백분율 오차(%)
        'MAE': float(np.mean(np.abs(t - p))),               # 평균 절대 오차(사이클)
        'RMSE': float(np.sqrt(np.mean((t - p) ** 2))),      # 제곱근 평균 제곱 오차(사이클)
    }


def cv_evaluate(model, df, feats, cv, target='log_life', group='policy'):
    """정책 단위 교차검증. 반환: (fold 평균 지표 dict, 셀별 CV 예측값 oof)"""
    X, y, g = df[feats].reset_index(drop=True), df[target].reset_index(drop=True), df[group].values
    oof = np.zeros(len(df))
    scores = []
    for tr, va in cv.split(X, y, groups=g):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        scores.append(metrics(y.iloc[va], oof[va]))
    s = pd.DataFrame(scores)
    out = {'MAPE': s.MAPE.mean(), 'MAPE_std': s.MAPE.std(), 'MAE': s.MAE.mean(), 'RMSE': s.RMSE.mean()}
    return out, oof


def fit_eval(model, train, test, feats, target='log_life'):
    """train으로 학습해서 test를 평가. 반환: (지표 dict, 학습된 모델, 예측값(log))"""
    m = clone(model).fit(train[feats], train[target])
    pred = m.predict(test[feats])
    return metrics(test[target], pred), m, pred
