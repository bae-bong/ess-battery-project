"""데이터 분할: Batch1을 정책(policy) 단위로 Train / Valid(Hold-out)으로 나눈다.
같은 충전 정책의 셀은 수명이 비슷해서, 셀 단위로 섞으면 성능이 과대평가된다."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

SEED = 42


def holdout_by_policy(df, test_size=0.2, seed=SEED):
    """정책 단위 Hold-out. 반환: (train_df, valid_df)"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df['policy']))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def group_cv(n_splits=5):
    """정책 단위 교차검증 (같은 정책이 train/valid fold에 갈라지지 않음)"""
    return GroupKFold(n_splits=n_splits)
