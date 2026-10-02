# ESS 배터리 수명 예측

> SKALA 울산캠퍼스 4반 · 데이터 분석 mini-Project (개인) · 배은빈

> 초기 100 사이클 데이터만으로 배터리 셀의 총 수명(cycle life)을 예측한다.

---

## 1. 프로젝트 개요

- **데이터셋**: MIT–Stanford 배터리 데이터 (Severson et al., 2019), Kaggle `itshpark/data-driven-prediction-of-battery-cycle`
- **학습**: Batch1 (2017-05-12, 46셀)
- **평가**: Batch2 (2018-02-20, 39셀, 필수) / Batch3 (2018-04-12, 44셀, 선택)
- **Task**: Regression, 타깃 `log10(cycle_life)`, 성능은 원 단위(사이클)로 환산해 **MAPE**로 보고
- **목표 성능**: 원논문 Regression 테스트 MAPE **9.1%**
- 수명(cycle_life)이 비어 있는 셀은 제외 (Batch2 8셀, Batch3 2셀)

## 2. 파일 구조

```text
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_EDA.ipynb          # EDA 5문항, 피처 탐색
│   └── 03_modeling.ipynb     # 모델 비교, 피처 ablation, 최종 평가, 오류 분석
├── src/
│   ├── preprocess.py         # .mat 로딩 → 셀 단위 피처 CSV 생성
│   ├── features.py           # 셀 1개 → 피처 1행 (초기 100 사이클)
│   ├── split.py              # 정책 단위 Hold-out / GroupKFold
│   ├── evaluate.py           # MAPE·MAE·RMSE, CV/평가 함수
│   └── train.py              # 최종 모델 학습 + 성능표 재현
├── results/                  # 성능표(csv), 그림
└── data/README.md            # 원본 데이터는 저장소에 포함하지 않음
```

> 피처 엔지니어링은 `01_EDA.ipynb`와 `src/features.py`에 있다.

## 3. 환경 설정 및 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 데이터: Kaggle에서 받은 .mat 3개를 data/ 에 둔다 (data/README.md 참고)
python -m src.preprocess     # data/features_all.csv 생성
python -m src.train          # 최종 성능표 출력 + results/model_performance_report.csv 저장
```

## 4. EDA 핵심 발견

- **Cycle Life 분포**: Batch1 중앙값 858.5 (534–1227), Batch2 472 (392–1186), Batch3 1005.5 (541–1935).
  단수명(<500) 비율은 Batch1 0%, Batch2 **72%**, Batch3 0%.
  → Batch1로 학습하면 Batch2의 대부분은 **학습 범위 밖**이며, 배치 간 분포 차이가 크다. 그래서 `log10` 타깃을 쓰고 배치별로 성능을 분리해 보고한다.
- **열화 곡선**: 초반 완만 → knee 이후 급락. knee는 수명의 약 **76–77%** 지점이다.
  → knee와 총 사이클 수는 수명을 알아야 계산되므로 **입력에서 제외(누수 방지)**.
- **ΔQ(V) = Q100(V) − Q10(V)**: `dq_var`와 log 수명의 상관은 **−0.846 / −0.905 / −0.787** (Batch1/2/3).
  → 초기 100 사이클 안에 수명 신호가 있고, 세 배치에서 방향이 같다. 핵심 피처로 채택.
- **C-rate vs 수명**: Batch1은 고속 충전일수록 수명이 짧지만(정책별 평균 수명 546–1226), Batch2·3에서는 이 관계가 사라지고 `chargetime_mean`·`tavg_mean`·`qd_slope`의 상관 부호도 배치마다 바뀐다.
  같은 `4.8C(80%)-4.8C` 정책의 평균 수명은 Batch1 753 / Batch2 484 / Batch3 1564로 다르다.
  → 충전·온도 피처는 보조로만 쓰고, 같은 정책의 셀이 학습·검증에 갈라지지 않게 **정책 단위로 분할**한다.
- **추가 확인 (VIF)**: `dq_var`(158), `dq_mean`(115), `dq_min`(101)은 서로 중복된다. `tavg_mean`과 `tmax_mean`도 중복이다.
  → 그룹마다 대표 1개만 사용하고, 규제 모델(Elastic Net)을 쓴다.

## 5. 모델링

### 5-1. 피처 엔지니어링 전략

- 모든 피처는 **2–100 사이클**에서만 계산한다 (사이클 1은 값이 0).
- **핵심**: `dq_var` = log10 Var[Q100(V) − Q10(V)] (`Qdlin` 기반 ΔQ의 분산)
- **보조(ablation 후보)**: `c1`, `chargetime_mean`, `tavg_mean`, `qd_slope`를 `dq_var`에 하나씩 추가해 효과 확인
- **제외**: `dq_mean`/`dq_min`/`tmax_mean`(중복), `dq_skew`/`dq_kurt`/`qd_2`/`ir_*`(상관 약함·불안정)
- **금지**: `knee`, 총 사이클 수 (데이터 누수)
- 전처리는 `StandardScaler`를 포함한 sklearn `Pipeline`으로 묶어, scaler도 학습 데이터로만 fit한다.

### 5-2. 모델 선택 및 근거

- **후보 모델**: 평균 예측(기준선 0), `dq_var` 단일 선형회귀(기준선 1), Elastic Net, Ridge, Random Forest, Gradient Boosting
- **피처 세트**: A = `dq_var` 단독 / 보조 피처를 하나씩 추가한 조합(ablation) 비교
- **최종 모델**: `StandardScaler → ElasticNet(alpha=0.003, l1_ratio=0.1)`, 피처 **A (`dq_var` 1개)**
- **선택 이유**
  1. 학습 셀이 46개뿐이라 단순하고 규제된 선형 모델이 유리하다. 로그 수명과 `dq_var`는 선형에 가깝다 (CV MAPE: 평균 예측 19.21% → `dq_var` 직선 7.23%).
  2. 선택은 **Batch1 정책 단위 CV 결과만으로** 했다. Batch2는 모델을 확정한 뒤 **한 번만** 평가했다.
  3. 충전·온도 피처를 하나씩 추가해 봤지만 CV 개선이 CV 표준편차(약 2.4%p)보다 훨씬 작았고(최대 −0.05%p), `qd_slope`는 오히려 +1.21%p 나빠졌다. 배치 간 상관 부호도 불안정해, 사전에 정한 규칙(차이가 CV 표준편차 안이면 단순한 모델 선택)에 따라 `dq_var` 단독을 채택했다.

<details>
<summary>피처 ablation 표 (펼치기)</summary>

| feature 조합             | Train(CV) MAPE | CV std | Valid MAPE | CV 변화(%p) |
| ------------------------ | -------------- | ------ | ---------- | ----------- |
| dq_var 단독 (최종 모델)  | 7.23           | 2.40   | 23.33      | 0.00        |
| + c1 (충전 속도)         | 7.31           | 2.50   | 23.44      | +0.08       |
| + chargetime_mean        | 7.18           | 2.15   | 23.16      | −0.05       |
| + tavg_mean (평균 온도)  | 7.44           | 2.27   | 23.28      | +0.21       |
| + qd_slope (용량 기울기) | 8.44           | 2.23   | 22.73      | +1.21       |

모델 후보 비교(Elastic Net/Ridge/RF/GB)는 `results/model_comparison_batch1.csv`, 전체 ablation은 `results/ablation_features.csv`에 있다. 모두 Batch1 정책 단위 CV 기준이다.

</details>

### 5-3. 데이터 분할

| 구분                    | 방법                                                                  |
| ----------------------- | --------------------------------------------------------------------- |
| Train (Batch1 CV)       | 정책 단위 `GroupKFold(5)` 평균 MAPE                                   |
| Valid (Batch1 Hold-out) | 정책 20%(5개 정책, 11셀)를 통째로 제외 (`GroupShuffleSplit`, seed 42) |
| Test (Batch2)           | Batch1 전체(46셀)로 재학습 → Batch2 평가                              |
| Test (Batch3, 선택)     | 같은 모델로 Batch3 평가                                               |

같은 충전 정책 셀(대부분 2개씩)이 학습·검증에 갈라지면 성능이 과대평가되므로, **CV와 Hold-out 모두 정책 단위**로 분할했다.

## 6. 성능 결과

| 구분                    | MAPE (%) | 비고                             |
| ----------------------- | -------- | -------------------------------- |
| Train (Batch1 CV)       | 7.23     | 정책 단위 5-fold 평균 (std 2.40) |
| Valid (Batch1 Hold-out) | 23.33    | 정책 5개, 11셀                   |
| Test (Batch2)           | 36.67    | MAE 177.5, RMSE 189.7 (사이클)   |
| Gap (Train-Valid)       | +16.10   | (+) : 과적합 의심                |
| Gap (Valid-Test)        | +13.34   | (+) : 배치간 일반화 저하 의심    |
| Gap (Target-Test)       | +27.57   | Target : 원논문 9.1%             |

**Batch3 추가 검증 (선택)**

| 구분                    | MAPE (%) | 비고                                             |
| ----------------------- | -------- | ------------------------------------------------ |
| Train (Batch1 CV)       | 7.23     |                                                  |
| Valid (Batch1 Hold-out) | 23.33    |                                                  |
| Test (Batch2)           | 36.67    |                                                  |
| Gap (Train-Valid)       | +16.10   | (+) : 과적합 의심                                |
| Gap (Valid-Test)        | +13.34   | (+) : 배치간 일반화 저하 의심                    |
| Gap (Target-Test)       | +27.57   | Target : 원논문 9.1%                             |
| Test (Batch3)           | 13.56    | MAE 170.5 (사이클)                               |
| Gap (Batch2-Batch3)     | −23.11   | Test 성능 간 비교, (−): Batch3가 Batch2보다 양호 |
| Gap (Target-Test)       | +4.46    | Batch 3 기준, 원논문 성능 비교                   |

- Gap = (뒤 구간 MAPE) − (앞 구간 MAPE), **(+)이면 성능이 나빠진 것**. Target–Test는 Test − 9.1.
- 참고(Batch2): 평균 예측 67.09%, Gradient Boosting(`dq_var`) 33.29%. 어떤 모델도 Batch2에서 9.1%에 근접하지 못했다. 최종 모델은 Batch2를 보기 전의 CV 기준으로 정했으므로 바꾸지 않았다.

### 해석

- **Train–Valid +16.1**: CV는 Batch1의 거의 모든 정책 범위를 학습에 쓴다. Valid는 정책 5개를 통째로 빼므로 학습 범위 밖의 극단 정책(수명이 매우 길거나 짧은 정책)이 섞여 오차가 커진다. 작은 표본(Valid 11셀)의 변동성도 있어 "과적합 의심"으로 읽되 과장하지 않는다.
- **Valid–Test +13.3**: 정책 단위로 분할해도 같은 배치 안이라는 한계가 있다. Batch2 오차는 **배치가 바뀌어 `dq_var`와 수명의 관계가 달라진 것**이 원인이다 (아래 오류 분석).
- **Target–Test +27.6**: 논문의 9.1%는 같은 배치 계열 안의 테스트 값이다. Batch1 → Batch2는 더 어려운 설정이라 직접 비교에는 한계가 있다. Batch3(13.56%)은 논문에 상당히 가깝다.

## 7. 오류 분석

![batch shift](results/batch_shift.png)

- **Batch2 오차 상위 10셀은 모두 과대 예측**이다 (+47% – +88%).
- **배치 이동(batch shift)**: 같은 `dq_var`에서도 Batch2 셀의 수명이 더 짧다. `log10(수명) = a + b·dq_var` 회귀선의 절편/기울기가 Batch1 2.124/−0.202, Batch2 1.604/−0.313, Batch3 1.577/−0.343으로 달라진다. Batch1에서 배운 직선을 그대로 쓰면 Batch2는 체계적으로 길게 예측된다.
- **예측 압축**: Batch2 예측은 540–1050인데 실제는 392–1186이다. 학습 범위(534–1227)에 맞춰 예측이 평균 쪽으로 쏠린다.
- **외삽이 주원인은 아니다**: Batch2 39셀 중 12셀은 `dq_var`가 Batch1 최댓값을 넘지만, 범위 안 27셀의 MAPE 36.7%와 범위 밖 12셀의 36.6%가 같다. 트리 모델(GB 33.3%)로 바꿔도 크게 줄지 않는다.
- **Spearman 순위상관 −0.716**: 순서 정보는 어느 정도 유지되지만, 수명 600 미만 구간에서는 −0.396으로 약하다.
- **원인 가설 및 개선 방향**
  - 가설: 제조 로트나 시험 조건이 배치마다 달라 `dq_var`–수명 관계가 이동했다. 이는 검증하지 않은 가설이며, 데이터에 로트 정보가 없어 확인할 수 없다.
  - 개선 방향 (1) 새 배치의 소수 셀(수 개)로 절편만 보정하는 재보정, (2) 여러 배치를 함께 학습하고 배치 단위로 검증, (3) 예측 구간(불확실성) 함께 제시.

## 8. ESS 도메인 해석

**이 모델로 어떤 BESS 의사결정을 지원할 수 있는가**

- **셀 조기 선별**: 수 개월짜리 수명 시험 대신 100 사이클 만에 장수명/단수명 후보를 가린다 (제조사 수율 선별, 입고 검사).
- **교체 시점 계획과 RUL 추정**: SOH 80% 교체 기준에 맞춰 교체 예산(CAPEX) 시점을 앞당겨 계획한다.
- **예지 보전, BMS/EMS 연계**: 수명이 짧을 것으로 예측된 셀·팩은 점검 주기를 줄이거나 운전 조건(충전 속도)을 보수적으로 조정하는 EMS 정책에 쓸 수 있다.
- `dq_var`(초기 100 사이클의 용량-전압 곡선 변화량)가 수명과 강하게 연결되는 것은 논문의 설명을 따른 해석이다. 같은 셀의 사이클 간 차이이므로 셀 고유 오프셋이 상쇄되어 배치 간에도 비교적 안정적일 것이라는 점은 본 분석에서 검증하지 않은 가설이다.

**한계와 실제 적용에 필요한 것**

- **배치 일반화 한계**: 새 배치에서 MAPE 36.7%(Batch2)까지 올라가므로 단독 판단 용도로는 부족하다. 선별 용도라도 새 배치마다 재보정이 필요하다.
- **오차 방향의 위험**: 오차가 모두 **과대 예측**이었다. 수명을 길게 잡으면 교체 시점이 늦어져 안전 측면에서 위험하므로, 보수적 보정이나 예측 구간이 필요하다.
- **적용 전 필요 조건**: (1) 새 배치 소수 셀로 재보정, (2) 배치 정보(로트, 시험 조건) 확보, (3) 불확실성 제시, (4) 수명 라벨 정의 확인 (일부 셀은 기록된 마지막 용량이 80% 기준과 맞지 않음).

## 9. 한계 및 향후 과제

- 학습 셀이 46개로 적고, 정책당 셀이 2개뿐이라 정책 단위 검증도 변동성이 크다 (Valid 11셀).
- Batch2는 단수명 셀이 72%로, Batch1 수명 범위(최소 534)와 거의 겹치지 않는다.
- 배치 이동의 원인(로트, 시험 조건, `newstructure` 셀 구조)은 데이터로 확인하지 못했다.
- 향후: 다중 배치 학습, 배치 재보정, 예측 구간, 초기 사이클 수(100)에 따른 민감도 분석.

## 10. 참고 문헌

- Severson, K. A. et al. _Data-driven prediction of battery cycle life before capacity degradation._ Nature Energy 4, 383–391 (2019).
- Dataset: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

## 11. 팀 구성 및 역할

- 배은빈 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2, Batch3)
