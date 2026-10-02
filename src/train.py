"""최종 모델 학습 + 과제 Reporting Format 성능표 재현.

실행: python -m src.train   (프로젝트 루트, data/features_all.csv 필요)
"""
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

from src.split import holdout_by_policy, group_cv
from src.evaluate import cv_evaluate, fit_eval

FEATS = ["dq_var"]          # 최종 피처 세트 A
TARGET_MAPE = 9.1           # 원논문 Regression MAPE (%)


def make_final_model():
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.003, l1_ratio=0.1, max_iter=20000),
    )


def run(path="data/features_all.csv", out="results/model_performance_report.csv"):
    df = pd.read_csv(path)
    b1n, b2n, b3n = sorted(df["batch"].unique())[:3]
    B1, B2, B3 = (df[df["batch"] == n].reset_index(drop=True) for n in (b1n, b2n, b3n))

    train, valid = holdout_by_policy(B1)       # 정책 단위 Hold-out
    cv = group_cv(n_splits=5)                  # 정책 단위 GroupKFold

    model = make_final_model()
    cv_s, _ = cv_evaluate(model, train, FEATS, cv)
    v_s, _, _ = fit_eval(model, train, valid, FEATS)
    t2, _, _ = fit_eval(model, B1, B2, FEATS)  # Batch1 전체로 재학습
    t3, _, _ = fit_eval(model, B1, B3, FEATS)

    tr, va, te2, te3 = cv_s["MAPE"], v_s["MAPE"], t2["MAPE"], t3["MAPE"]
    rows = [
        ("Train (Batch1 CV)", tr, ""),
        ("Valid (Batch1 Hold-out)", va, ""),
        ("Test (Batch2)", te2, ""),
        ("Gap (Train-Valid)", va - tr, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", te2 - va, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", te2 - TARGET_MAPE, "Target : 원논문 9.1%"),
        ("Test (Batch3)", te3, ""),
        ("Gap (Batch2-Batch3)", te3 - te2, "Test 성능 간 비교"),
        ("Gap (Target-Test, Batch3)", te3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"),
    ]
    report = pd.DataFrame(rows, columns=["구분", "MAPE(%)", "비고"])
    report["MAPE(%)"] = report["MAPE(%)"].round(2)
    print(report.to_string(index=False))
    report.to_csv(out, index=False)
    return report


if __name__ == "__main__":
    run()