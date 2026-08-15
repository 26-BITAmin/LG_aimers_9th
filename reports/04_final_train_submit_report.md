# 04. 전체 학습·제출 패키지 및 첫 리더보드 결과

## 결론

- 실행 노트북의 코드 셀 7개가 모두 정상 완료됐고 오류 셀은 없다.
- 2019~2024 전체 학습 데이터로 기존 02 LightGBM과 03 CatBoost를 재학습했다.
- 최종 예측은 **LightGBM 65% + CatBoost 35%의 logit 평균**이다.
- 저장 전후 예측, 행 셔플, 행 분할, 단일 행 추론의 최대 절대 오차가 모두 0이었다.
- 제출 ZIP은 요구된 5개 파일만 포함하고 압축 손상 검사와 평가용 `script.py` 실행을 통과했다.
- 첫 제출은 2026-08-14 23:11:07에 정상 평가됐으며 **Public Score 856.9040639322**, 추론 시간 6초를 기록했다.
- 이 점수는 제출 파이프라인의 정상 작동을 확인한 첫 앵커다. 팀 챔피언 987.4136과 현재 상위권에는 미치지 못하므로 다음 단계는 피처 ablation, 확률 보정, 팀 OOF 앙상블이다.

## 입력·실행 무결성

- 실행 환경: Colab A100-SXM4 40GB
- Python: 3.12.13
- NumPy: 2.0.2
- pandas: 2.2.2
- joblib: 1.5.3
- LightGBM: 4.6.0
- CatBoost: 1.2.8
- 사용 CPU thread: 8
- 학습 전체 소요 시간: 79.35초
- 노트북 실행 셀: 7/7
- 오류 셀: 0

입력과 실행 결과의 SHA-256은 다음과 같다.

| 파일 | SHA-256 |
|---|---|
| `train.csv` | `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff` |
| 실행 완료 `04_final_train_submit.ipynb` | `4743afb96e3bd910d7250b93f8787cc37b871e4aff12d817e53fd7c9430765cf` |
| `training_summary.json` | `6f7c8d78787b801b469f99e5b8e18bf670ebe8fe77a5234740201c9986705dca` |
| `lgbm_feature_importance.csv` | `754f7b1a6d82b13116ae50c22e02f907a4efcf8154dd85c17eb50e9375085f79` |
| `catboost_feature_importance.csv` | `59c6990bc0816adfbe596d9586694de1076e58e450625dee1c2a55d24d307f57` |
| 제출 ZIP | `55202031ecd64d6ab1d5546d068700c4dcf3e715e57aa82cff71f0ea605e9e0c` |
| 첫 제출 결과 화면 | `6330447f961f62bd8ed26c9b2314e88b6a2a4189aa093dfb92acf0792b0920b1` |

Colab에서 읽은 `train.csv`의 해시는 로컬 원본과 일치한다. 학습 행 수는 1,475,092이고 시즌은 2019~2024가 정확히 존재한다.

## 최종 모델 구성

| 항목 | LightGBM | CatBoost |
|---|---:|---:|
| 학습 행 | 1,314,088 | 1,475,092 |
| 학습 경기 유형 | `R`만 | 전체 `R/F` |
| 최근성 half-life | 2.5년 | 0.5년 |
| 기준 시즌 | 2025 | 2025 |
| 피처 수 | 66 | 66 |
| 범주형 피처 수 | 9 | 10 |
| 선수 ID | 제외 | 투수·타자 ID 포함 |
| `season` 피처 | 포함 | 제외 |
| `game_type` 피처 | 형식상 포함 | 제외 |
| iteration/tree | 500 | 85 |
| 학습 장치 | CPU | GPU |
| 학습 시간 | 38.48초 | 8.08초 |

LightGBM은 `R` 행만 학습했으므로 `game_type` 값이 항상 `R`이고 실제 gain importance가 0이다. CatBoost는 입력에서 `game_type`을 제거했다. 따라서 최종 앙상블은 실질적으로 `game_type` 값을 예측 신호로 사용하지 않는다. 이는 `game_type` 피처 제거와 `F` 학습 행 제외가 서로 다른 선택임을 의미한다.

### LightGBM 고정 설정

- `learning_rate=0.03`
- `num_leaves=31`
- `min_child_samples=1000`
- `subsample=0.9`
- `colsample_bytree=0.85`
- `reg_alpha=0.1`
- `reg_lambda=2.0`
- 선수 ID 제외
- `R` 행만 학습

### CatBoost 고정 설정

- `iterations=85`
- `learning_rate=0.04`
- `depth=8`
- `l2_leaf_reg=12`
- `random_strength=0.5`
- `max_ctr_complexity=1`
- `season`, `game_type` 제외
- 선수 ID 포함

CatBoost 85회는 2025와 가장 가까운 2024 forward fold의 best iteration을 사용한 값이다. 2023 best iteration은 11이었으므로 iteration의 시간 안정성은 후속 실험에서 다시 확인해야 한다.

## 모델 선택 근거

최종 학습 자체는 검증 점수를 새로 계산하지 않는다. 아래 값은 03에서 고정한 2023·2024 forward OOF 근거다.

| 모델 | 2023·2024 pooled Brier | BSS |
|---|---:|---:|
| 02 LightGBM | 0.248602074 | 539.23 |
| 03 CatBoost | 0.248943385 | 402.68 |
| probability 65:35 | 0.248449856 | 600.13 |
| **logit 65:35** | **0.248449426** | **600.30** |

정밀 최적 logit 비중은 64.08:35.92였지만 고정 65:35와 BSS 차이가 0.04뿐이었다. 비중 과적합을 줄이기 위해 고정 65:35를 배포했다. 한 시즌에서 선택한 비중을 다른 시즌에 적용한 교차 적합 BSS도 595.97로 기존 LightGBM보다 높았다.

## 자동 재현·행 독립성 검사

| 검사 | 최대 절대 오차/결과 |
|---|---:|
| LightGBM native 저장 후 재로딩 | 0.0 |
| CatBoost native 저장 후 재로딩 | 0.0 |
| 입력 행 셔플 후 원순서 복원 | 0.0 |
| 입력 행을 여러 청크로 분할 | 0.0 |
| 한 행씩 개별 추론 | 0.0 |
| 감사 표본 행 수 | 64 |
| 패키지 `script.py` 반환값 | 0 |

제출 스크립트에는 test 행을 이용한 `groupby`, 빈도값, rolling, expanding, target encoding 또는 전체 test 평균 보정이 없다. 피처의 행 방향 평균은 같은 행의 직전 1·3·5경기 공식 `asof_*` 컬럼을 묶는 연산이며 다른 test 행을 참조하지 않는다.

배포본의 5행 형식 확인용 test에서 생성된 확률 범위는 다음과 같다.

- 최솟값: 0.4195186894
- 평균: 0.4640251093
- 최댓값: 0.5125655028

이 5행은 실제 평가 데이터가 아니므로 분포나 성능 판단에 사용하지 않는다. 오직 입출력 형식과 실행 성공 확인에만 사용한다.

## 제출 패키지

실제 제출 시 파일명은 DACON의 30자 제한을 맞추기 위해 `submit_01.zip`으로 변경했다. 파일명만 변경했으므로 ZIP 바이트와 SHA-256은 생성본과 동일하다.

```text
submit_01.zip
├── model/
│   ├── lgbm_model.txt
│   ├── catboost_model.cbm
│   └── artifacts.joblib
├── script.py
└── requirements.txt
```

| ZIP 구성원 | 원본 크기 | SHA-256 |
|---|---:|---|
| `model/lgbm_model.txt` | 1,778,825 bytes | `94d87f291f74e83a16c80e99cb096ce9c9fd06ef902f94238ee9115384e73377` |
| `model/catboost_model.cbm` | 405,268 bytes | `859cc4d54c16ce6f1b35c253d171277e2025169c3790d34ee52793879d754fbe` |
| `model/artifacts.joblib` | 1,217 bytes | `7b69cffa08eeb5f7bb05b2331efd7aac08cfb5d4b86a1d6decdb9cce2aa94cd7` |
| `script.py` | 12,013 bytes | `13ebd43a730cad521de156dd66fb2446595c5d010f39739234aa97895a796cc4` |
| `requirements.txt` | 171 bytes | `193242d69e0e3f5d4992ca7d2470ee33115aeb05c733571b4034cbb397b71ee5` |

- ZIP 압축 크기: 1,005,550 bytes, 약 0.96 MiB
- ZIP 구성원: 정확히 5개
- ZIP 손상 검사: 통과
- LightGBM tree 수: 500
- LightGBM feature 수: 66
- CatBoost native magic: `CBM1`
- 제출용 추가 패키지: LightGBM 4.6.0, CatBoost 1.2.8

## 첫 리더보드 제출

| 항목 | 결과 |
|---|---:|
| 제출 파일명 | `submit_01.zip` |
| 제출 시각 | 2026-08-14 23:11:07 |
| Public Score | **856.9040639322** |
| 추론 시간 | 6초 |
| 실행 상태 | 정상 완료 |

Public 856.90은 OOF 600.30보다 높지만 두 점수를 직접 비교할 수 없다. BSS 분모인 실제 성공률 `r × (1-r)`와 행 구성이 다르기 때문이다. Public 상승만으로 OOF보다 일반화 성능이 좋아졌거나 누수가 있다고 해석하지 않는다.

2026-08-15 확인 당시 공개 리더보드 1위는 약 1,176.55, 100위는 약 1,083.03이었다. 본선권 진입에 만족하지 않고 **팀 최종 목표를 최소 1,165점 이상**으로 고정한다. 이 목표까지 우리 첫 제출 856.9041은 약 308.10점, 팀 챔피언 987.4136은 약 177.59점의 추가 개선이 필요하다. 1,100점과 1,120점은 각각 중간 진행 체크포인트일 뿐 최종 성공 기준으로 사용하지 않는다.

## 피처 중요도 진단

### LightGBM 상위 피처

| 순위 | 피처 | gain importance |
|---:|---|---:|
| 1 | `asof_pitcher_success_rate` | 62,334.26 |
| 2 | `smoothed_pitcher_success_rate` | 36,948.45 |
| 3 | `pitcher_team_id` | 25,953.40 |
| 4 | `batter_team_id` | 24,903.59 |
| 5 | `count_state` | 24,604.04 |
| 6 | `asof_pitcher_reverse_rate` | 20,642.71 |
| 7 | `season` | 14,317.44 |
| 8 | `hand_matchup` | 14,155.49 |
| 9 | `asof_pitcher_ball_rate` | 13,186.35 |
| 10 | `asof_pitcher_prev5_game_success_rate` | 12,053.76 |

상위 5개가 전체 gain의 42.74%, 상위 10개가 60.92%를 차지한다. `smoothed_pitcher_success_rate`, `count_state`, `hand_matchup`, `pitcher_recent_success_mean`처럼 직접 만든 행 단위 피처가 상위권이므로 추가 피처 ablation의 근거가 있다.

### CatBoost 상위 피처

| 순위 | 피처 | importance |
|---:|---|---:|
| 1 | `hand_matchup` | 11.7431 |
| 2 | `batter_team_id` | 6.5890 |
| 3 | `asof_pitcher_success_rate` | 6.0939 |
| 4 | `asof_pitcher_ball_rate` | 6.0852 |
| 5 | `asof_pitcher_prev5_game_success_rate` | 5.1119 |
| 6 | `asof_pitcher_reverse_rate` | 4.6743 |
| 7 | `pitcher_recent_success_mean` | 4.4493 |
| 8 | `pitcher_team_id` | 4.4322 |
| 9 | `asof_batter_success_rate` | 3.7912 |
| 10 | `pitcher_id` | 3.7019 |

상위 5개가 전체 importance의 35.62%, 상위 10개가 56.67%를 차지한다. 투수 ID는 일부 신호가 있지만 타자 ID 중요도는 0이었다. 다음 실험에서는 `투수 ID만`, `투타 ID 모두`, `ID 없음`을 분리해 확인할 필요가 있다.

두 모델의 중요도 순위 상관은 약 0.824이고 상위 10개 교집합은 7개다. 중요도 정의가 서로 달라 절대값을 직접 비교할 수는 없지만, CatBoost가 손잡이와 투수 ID를 더 강하게 이용하는 차이는 모델 다양성의 한 근거다.

## 현재 한계

1. 최종 학습 단계에는 정답이 있는 validation이 없으므로 04 자체의 새 검증 점수는 없다.
2. CatBoost iteration 85는 최신 2024 fold를 기준으로 고정했으며 시즌별 최적 iteration 변동이 크다.
3. 현재 21개 행 단위 파생 피처를 사용하지만 비율 차이, 최근 폼 변동성, 구종 엔트로피, 상황 상호작용은 아직 없다.
4. `trackman_history.csv`는 현재 모델에서 전혀 사용하지 않았다.
5. Public 리더보드는 테스트 100%이지만 반복 제출로 후보를 선택하면 같은 비공개 정답에 과적합할 수 있다.
6. 팀 챔피언과 우리 모델의 OOF를 직접 결합한 팀 수준 앙상블은 아직 수행하지 않았다.

## Trackman 사용 가능성 사전 점검

`trackman_history.csv`는 2019~2024의 1,793,078행이며 현재 투구의 정답과 1:1로 연결되는 테이블이 아니다. 기초 점검 결과는 다음과 같다.

| 항목 | 메인 데이터 | Trackman | 직접 ID 교집합 |
|---|---:|---:|---:|
| 투수 | 792명 | 906명 | 0명 |
| 타자 | 830명 | 913명 | 0명 |

따라서 `pitcher_id == pitcher_trackman_id` 또는 `batter_id == batter_trackman_id` 방식으로 병합하면 안 된다. 팀도 메인 데이터는 익명 숫자 ID, Trackman은 문자열 코드이므로 검증 없는 직접 매핑은 불가능하다.

Trackman 주요 물리량의 비결측률은 모두 약 99.3~99.6%다. 구종군 행 수는 fastball 931,120, breaking 512,851, offspeed 326,809, other 22,298이다. 데이터 자체의 품질과 크기는 충분하지만 ID 연결이 핵심 병목이다.

### 1단계: 공식 데이터 내부 매핑 가능성 검증

외부 선수 명단이나 비공개 정보는 사용하지 않는다. 다음 공식 피처의 시계열 지문만 이용해 메인 익명 투수와 Trackman 투수의 후보 대응을 탐색한다.

- 시즌별 투수 손잡이
- 시즌별 팀 소속 패턴
- fastball·breaking·offspeed 사용 비율
- 시즌 간 구종 구성 변화
- 표본 수와 활동 시즌 패턴

메인 데이터의 마지막 `asof_pitcher_*` 구종 비율과 Trackman의 시즌 누적 구종 비율을 비교할 수 있다. 단일 시즌 최단거리만으로 매핑하지 않고 여러 시즌에서 같은 일대일 대응이 반복되는지 확인해야 한다. 후보 간 거리 차이가 작거나 시즌별 매핑이 충돌하는 선수는 미매핑으로 남긴다.

### 2단계: 누수 없는 Trackman 집계

매핑이 충분히 안정적인 선수에게만 다음 피처를 만든다.

- 구종군별 사용 비율과 엔트로피
- 구속 평균·표준편차·상위/하위 분위수
- 회전수와 구속의 조합
- 수직·수평 무브먼트 평균과 변동성
- extension, release height, release side 평균과 변동성
- 구종군 사이 구속·무브먼트 분리도
- Trackman 표본 수와 결측 여부

시간 누수를 막기 위해 validation 시즌 `s`의 행에는 원칙적으로 `s-1` 시즌까지의 Trackman만 사용한다. 2025 실제 추론에는 2019~2024 전체 Trackman을 사용한다. 같은 시즌의 월별 로그를 쓰는 방법은 2025 Trackman이 없어서 train/test 피처 생성 조건이 달라지므로 첫 실험에서는 제외한다.

### 3단계: 매핑 실패 시 대안

선수 매핑이 불안정하면 억지로 선수별 Trackman 피처를 사용하지 않는다. 대신 이전 시즌까지의 다음 저해상도 집계만 검토한다.

- 투수 손잡이별 리그 구속·무브먼트 기준값
- 구종군별 리그 평균과 시즌 변화
- 신뢰 가능한 팀 매핑이 별도로 성립할 때만 팀·손잡이별 집계

저해상도 집계는 행마다 값의 차이가 작아 효과가 제한적일 수 있다. 이 경우 Trackman을 포기하는 것이 잘못된 선수 매핑보다 안전하다.

## 다음 실험

1. `05_feature_ablation_cv.ipynb`에서 `game_type`을 제거한 채 현재 파생 피처를 기준선으로 고정한다.
2. 비율 차이, 최근 폼 변동성, 구종 구성, 상황 상호작용, 투수팀 기대 승률을 묶음별로 추가한다.
3. 2022·2023·2024 forward CV를 실행하고 2023·2024 pooled Brier/BSS와 개별 시즌 방향을 함께 본다.
4. 승리한 피처 세트에 대해서만 학습 범위, half-life, 선수 ID, 모델 파라미터를 최적화한다.
5. 팀 챔피언의 동일 OOF를 받아 row 단위 예측 상관과 교차 적합 혼합비를 구한다.
6. Trackman 선수 매핑은 별도 실험으로 분리하고, 다년간 일대일 안정성이 확인되기 전에는 모델 피처로 사용하지 않는다.
7. 리더보드는 하루 1~2회만 사용하고 CV를 통과한 후보의 전이성 확인에만 이용한다.

## 최종 판정

1. 04 학습과 제출 패키지는 기술적으로 정상이며 그대로 재현 가능하다.
2. 첫 Public 856.9041은 파이프라인 앵커로 기록한다.
3. 현재 모델은 최종 본선권 목표에는 부족하므로 챔피언으로 고정하지 않는다.
4. 다음 우선순위는 `game_type` 복원이 아니라 누수 없는 행 단위 피처 확장이다.
5. Trackman은 잠재력이 크지만 ID 체계가 분리돼 있어 매핑 검증이 선행되어야 한다.
6. 개인 모델 점수뿐 아니라 팀 챔피언과의 OOF 앙상블 개선량을 최종 의사결정 기준으로 사용한다.
7. 최종 성공 기준은 Public/종료 시점 Private **1,165점 이상**이며, 본선권 추정선 통과만으로 실험을 중단하지 않는다.

## 공식 확인 링크

- 평가 산식·코드 제출 환경: <https://dacon.io/competitions/official/236743/overview/evaluation>
- 제출 횟수·행 독립성·외부 데이터 규칙: <https://dacon.io/competitions/official/236743/overview/rules>
- 2026-08-15 공개 리더보드: <https://dacon.io/competitions/official/236743/leaderboard>
