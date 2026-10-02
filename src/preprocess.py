"""원본 .mat -> data/features_all.csv 생성 (python -m src.preprocess)"""
import gc, os
import mat73
import numpy as np
import pandas as pd
import scipy.io as sio
from src.features import extract

DATA_DIR = 'data'
BATCH_FILES = {
    'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}


def load_mat(path):
    try:
        return mat73.loadmat(path)          # MATLAB v7.3
    except Exception:
        return sio.loadmat(path, simplify_cells=True)


def to_list_of_dicts(d):
    if isinstance(d, dict):
        keys = list(d.keys())
        n = len(d[keys[0]])
        return [{k: d[k][i] for k in keys} for i in range(n)]
    return d


def process_batch(cells, bname):
    rows, skipped = [], []
    for i, cell in enumerate(cells):
        cl = cell.get('cycle_life')
        if cl is None or not np.isfinite(np.asarray(cl, dtype=float).ravel()[0]):
            skipped.append(i); continue
        out = extract(cell, bname, i)
        if out is None:
            skipped.append(i); continue
        rows.append(out[0])
    print(f'[{bname}] 사용 {len(rows)}개, 제외 {len(skipped)}개 (cell_id): {skipped}')
    return rows


def build(out_path=os.path.join(DATA_DIR, 'features_all.csv')):
    rows = []
    for bname, fname in BATCH_FILES.items():
        m = load_mat(os.path.join(DATA_DIR, fname))
        cells = to_list_of_dicts(m['batch'])
        rows += process_batch(cells, bname)
        del m, cells; gc.collect()
    feat = pd.DataFrame(rows)
    feat['log_life'] = np.log10(feat['cycle_life'])
    feat.to_csv(out_path, index=False)
    print('저장:', out_path, feat.shape)
    return feat


if __name__ == '__main__':
    build()
