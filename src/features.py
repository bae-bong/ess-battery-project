"""배터리 셀 1개에서 초기(2~100 사이클) feature를 뽑는 함수 모음.
수명(cycle_life)·knee는 정답 정보이므로 입력 feature에는 쓰지 않는다 (data leakage 방지)."""
import re
import numpy as np
from scipy import stats

EPS = 1e-12
N_EARLY, N_LATE = 10, 100   # ΔQ(V) = Q100(V) - Q10(V)
FIRST, LAST = 2, N_LATE + 1   # summary 배열 구간 [2, 101) - 원본 EDA 노트북과 동일 (검증 완료)


def get_field(cell, key, k=None):
    """cell['cycles'][key][k] (mat73 dict-of-lists / list-of-dicts 둘 다 대응)"""
    cyc = cell['cycles']
    if isinstance(cyc, dict):
        v = cyc[key]
        return v if k is None else v[k]
    return cyc[k][key]


def to_arr(x):
    return np.asarray(x, dtype=float).ravel()


def first_crate(policy):
    """'8C(35%)-3.6C' -> 8.0 (정책 문자열의 첫 C-rate)"""
    m = re.match(r'\s*([\d.]+)C', str(policy))
    return float(m.group(1)) if m else np.nan


def knee_cycle(qd, frac=0.95):
    """EDA 참고용. 입력 feature로 쓰지 않는다(leakage)."""
    qd = to_arr(qd)
    ref = np.nanmedian(qd[2:12])
    below = qd < frac * ref
    for i in range(2, len(qd) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            return i
    return np.nan


def extract(cell, bname, i):
    """셀 1개 -> (feature 한 줄 dict, ΔQ 곡선, Qd 곡선). 못 만들면 None."""
    s = cell['summary']
    qd = to_arr(s['QDischarge']); ir = to_arr(s['IR'])
    tavg = to_arr(s['Tavg']); tmax = to_arr(s['Tmax']); ct = to_arr(s['chargetime'])
    if len(qd) <= N_LATE:
        return None
    q_late = to_arr(get_field(cell, 'Qdlin', N_LATE))
    q_early = to_arr(get_field(cell, 'Qdlin', N_EARLY))
    dq = q_late - q_early
    dq = dq[np.isfinite(dq)]
    if len(dq) < 10:
        return None

    cyc = np.arange(len(qd))
    sl = slice(FIRST, LAST)
    ok = (qd[sl] > 0.5) & (qd[sl] < 1.5)
    qd_slope = np.polyfit(cyc[sl][ok], qd[sl][ok], 1)[0] if ok.sum() > 2 else np.nan
    ctm = ct[sl]; ctm = ctm[(ctm > 0) & (ctm < 60)]

    policy = str(cell.get('policy_readable') or cell.get('policy') or 'unknown')
    row = dict(
        batch=bname, cell_id=i, cycle_life=float(cell['cycle_life']),
        policy=policy, c1=first_crate(policy),
        dq_min=np.log10(abs(dq.min()) + EPS),
        dq_var=np.log10(np.var(dq) + EPS),
        dq_mean=np.log10(abs(dq.mean()) + EPS),
        dq_skew=np.log10(abs(stats.skew(dq)) + EPS),
        dq_kurt=np.log10(abs(stats.kurtosis(dq)) + EPS),
        qd_2=qd[FIRST], qd_slope=qd_slope,
        ir_2=ir[FIRST], ir_diff=ir[N_LATE] - ir[FIRST],
        tavg_mean=np.nanmean(tavg[sl]), tmax_mean=np.nanmean(tmax[sl]),
        chargetime_mean=np.nanmean(ctm) if len(ctm) else np.nan,
        knee=knee_cycle(qd),
    )
    return row, dq, qd
