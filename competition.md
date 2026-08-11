# competition.md — 대회 정보 및 규칙

> LG Aimers 9기 : 투구 제구 성공 확률 예측 AI 온라인 해커톤 (Dacon)
> 이 문서는 **대회 측이 정한 사실**만 담는다. 우리 분석 결과는 `EDA_정리.md` 참조.

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| 주최 | LG Aimers (Dacon 운영) |
| 기간 | 2026.08.05 ~ 2026.09.02 09:59 |
| 과제 | 투구 직전 정보만으로 각 투구의 **제구 성공 확률** 예측 |
| 형식 | **코드 제출 대회** (`submit.zip` 업로드, 평가 서버가 실행) |
| 다음 단계 | Private 상위 약 100명 → 오프라인 해커톤 Phase 3 (1박 2일) |

### 제구 성공/실패 정의

각 투구의 공 위치를 기준으로 하며, 아래 **3가지 중 하나라도** 해당하면 실패(`control_success=0`):

1. 스트라이크존 가운데 부근으로 들어간 공
2. 스트라이크존에서 크게 벗어난 공
3. 포수의 요구 방향과 반대로 들어간 공

그 외 유효한 투구가 성공(`control_success=1`).

> 주의: 세 유형은 **상호배타적이 아니다.** 1번과 3번이 동시에 발생하는 투구가 전체의 3.41% 존재.

---

## 2. 평가

### 산식 — Brier Skill Score

```
Score = max(0, 100000 × (1 - Brier Score / (r × (1-r))))

Brier Score = mean((p_i - y_i)^2)
r = 평가 데이터 전체의 평균 제구 성공률 (비공개)
p_i = i번째 샘플의 제구 성공 예측 확률
y_i = i번째 샘플의 실제 정답 (0 또는 1)
```

- Public Score = 전체 테스트 데이터 100%
- Private Score = 대회 종료 시점의 Public Score (동일)

### 기준선

| 항목 | 값 |
|---|---|
| 수료 조건 | Phase 1 이수 + Phase 2 Public Score **549.51 이상** |
| 549.51의 출처 | 운영진 베이스라인 추론 코드를 운영진 평가 환경에서 실행한 점수 |
| 1차 평가 | 리더보드 Private Score 100% |
| 2차 평가 | Phase 3 진출 희망 팀은 코드 + PPT 제출 후 검증 |

### 환산 계수 (실무용)

`r ≈ 0.49` 기준 `∂Score/∂BS = -10^5 / (r(1-r)) ≈ -400,000`
→ **ΔBrier 1e-4 = 약 40점**. 모든 실험 결과는 이 환산으로 기록할 것.

---

## 3. 데이터

### 배포 구조

```
open.zip
├── baseline_submit.zip            # 베이스라인 제출 파일 예시
├── data/
│   ├── train.csv                  # 1,475,092행 × 49컬럼 (정답 포함)
│   ├── test.csv                   # 배포본은 5행 샘플. 실제 245,789행 × 48컬럼
│   ├── sample_submission.csv      # 5행 × 2컬럼 샘플
│   └── trackman_history.csv       # 1,793,078행 × 30컬럼 (2019~2024)
└── data_description.md
```

- `train.csv` = **2019~2024 시즌**, `test.csv` = **2025 시즌** (분석으로 확정)
- 실제 `test.csv`는 평가 서버에서 동일 경로·동일 컬럼 구조로 교체됨

### train / test 컬럼 (48개 입력 + 1개 정답)

**식별자 및 경기 정보**

| 컬럼 | 설명 |
|---|---|
| `row_id` | 샘플 고유 식별자 (제출 매칭용) |
| `season` | 시즌 연도 |
| `game_month` | 경기 월 |
| `game_dayofweek` | 경기 요일 (월=0, 일=6) |
| `inning` | 투구 직전 이닝 |
| `top_bottom` | `T`=초, `B`=말 |
| `game_type` | 경기 유형 코드 (`R`, `F`) |

**카운트 및 점수**

`balls_before`, `strikes_before`, `outs_before`, `run_top_before`, `run_bot_before`,
`run_total_before`, `score_diff_home`, `score_diff_pitcher_team`

**주자 및 상황 중요도**

`runner_on_1b`, `runner_on_2b`, `runner_on_3b`, `num_runners_on`,
`base_state` (`___`, `1__`, `_2_`, `__3`, `12_`, `1_3`, `_23`, `123`),
`home_win_expectancy`, `away_win_expectancy` (0~100), `li`

**선수 및 팀**

`pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id`

**투구 직전 기준 과거 이력 (`asof_*` 19개)**

투수: `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`,
`asof_pitcher_middle_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`,
`asof_pitcher_prev1/3/5_game_success_rate`, `asof_pitcher_prev1/3/5_game_middle_rate`,
`asof_pitcher_pitchmix_n`, `asof_pitcher_fastball_rate`, `asof_pitcher_breaking_rate`,
`asof_pitcher_offspeed_rate`

타자: `asof_batter_n`, `asof_batter_success_rate`, `asof_batter_middle_rate`

> `asof_*`는 해당 투구 **직전까지** 확인 가능한 과거 기록으로 사전 계산됨.
> 현재 투구 이후 확정 정보는 포함되지 않음. 표본 수 0이면 rate 컬럼은 결측.

**정답**

| 컬럼 | 설명 |
|---|---|
| `control_success` | `1`=제구 성공, `0`=제구 실패 |

### `trackman_history.csv` (30컬럼)

2019~2024 Trackman 로그. **2025 데이터는 제공되지 않음.**
main 데이터와 **1:1 결합되는 테이블이 아니며** ID 공간이 다르다
(`pitcher_trackman_id` ≠ `pitcher_id`, `trackman_game_id` ≠ `row_id`).

`trackman_id`, `season`, `game_date`, `game_month`, `game_dayofweek`, `trackman_game_id`,
`pitch_no`, `inning`, `top_bottom`, `balls_before`, `strikes_before`, `outs_before`,
`pitch_of_pa`, `pitcher_trackman_id`, `batter_trackman_id`, `pitcher_hand`, `batter_hand`,
`pitcher_team`, `batter_team`, `tagged_pitch_type`, `auto_pitch_type`, `pitch_type_group`,
`rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`,
`rel_side`, `zone_speed`

### `sample_submission.csv`

| 컬럼 | 설명 |
|---|---|
| `row_id` | 평가 데이터 `test.csv`의 샘플 식별자 |
| `control_success` | 제구 성공 확률, **0 이상 1 이하 실수** |

제출 시 `row_id` 값이 평가 서버 `test.csv`와 **정확히 일치**해야 한다.

---

## 4. 규칙 — 금지 사항

### 4-1. 사전학습 모델 및 가중치

- 공개 배포되고 **최소한 비상업적 이용이 허용된 라이선스**(MIT, Apache 2.0 등)만 사용 가능
- 조건 미충족 모델/가중치 사용 불가

### 4-2. 외부 API

- 원격 서버 기반 API 모델(OpenAI API, Gemini API 등) **사용 불가**
- 모든 작업은 로컬에서 코드로 실행·재현 가능해야 함

### 4-3. 외부 데이터

- Phase 2 제공 공식 데이터 외 **외부 데이터 사용 금지**

### 4-4. 평가 데이터 예측 원칙 ★

평가 데이터의 각 행은 **독립적으로** 예측해야 한다.
평가 서버에서 `test.csv` 전체가 주어져도, 다른 행을 이용해 현재 행의 피처를 만들 수 없다.

**금지 예시**

- `test.csv` 내부 행들을 이용한 선수별·팀별·월별 누적 통계
- `test.csv` 내부 빈도값 또는 분포 통계
- `test.csv` 내부 target encoding
- `test.csv` 행 순서 기반 rolling / expanding feature
- 평가 데이터 전체를 보고 만든 **사후 보정값**

**허용**

- 운영 측이 제공한 `asof_*` 컬럼 (각 행의 투구 직전까지 계산된 공식 입력 피처)
- 공식 `train.csv`로 만든 집계를 test 행의 **자기 자신 피처**로 lookup

### 4-5. 사용 금지 정보

- 현재 투구 **이후**에 확정되는 모든 정보
- 현재 투구의 실제 **위치 또는 코스**
- 현재 투구의 실제 **판정, 결과, 제구 성공 여부**
- 현재 투구의 실제 **구종**
- 현재 투구의 **Trackman 측정값**
- **2025년 Trackman 데이터**
- 평가 데이터 내부 다른 행 기반 누적/빈도/분포/rolling/target encoding

---

## 5. 제출

### 파일 구조

```
submit.zip
├── model/              # 모델 가중치 등 (예: model.pt, *.pkl, *.parquet)
├── script.py           # 평가 서버가 자동 실행하는 추론 코드
└── requirements.txt    # pip install -r 로 설치 가능한 형태
```

**최상위 폴더가 추가로 들어가면 설치 오류.** 디렉토리명·파일명 완전 일치 필수.

### 평가 서버가 자동 추가하는 항목

```
├── data/                       # 평가 데이터 (읽기 전용)
└── output/submission.csv       # script.py 실행 결과가 저장될 경로
```

`script.py`는 반드시 `./output/submission.csv`를 생성해야 한다.

### 제한

| 항목 | 제한 |
|---|---|
| zip 용량 | ≤ 10GB (압축 해제 후 ≤ 32GB) |
| 패키지 설치 시간 | ≤ 10분 (초과 시 설치 오류) |
| 추론 실행 시간 | **≤ 10분 (245,789행)** |
| 인터넷 | 패키지 설치 외 **차단** |

> 라이브러리 설치 에러는 설치 오류로 처리되며 **일일 제출 횟수에 반영되지 않는다.**

### 평가 서버 사양

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA L4 (VRAM 22.4 GiB) |
| CPU | 6 vCPU |
| RAM | 28 GB |
| Python | 3.11.15 |
| CUDA | 12.8 |

### 기본 설치 패키지 — **requirements.txt에 넣지 말 것**

```
torch==2.7.1+cu128      pandas==2.0.3         numpy==1.26.4
scipy==1.15.3           scikit-learn==1.8.0   joblib==1.5.3
threadpoolctl==3.6.0    narwhals==2.21.2      transformers==4.46.3
accelerate==1.9.0       sentencepiece==0.1.99 regex==2023.12.25
tqdm==4.66.4            loguru==0.7.2         pyyaml==6.0.1
rich==13.7.1
```

버전이 다른 걸 명시하면 설치 에러가 날 수 있다. **GBDT 계열(lightgbm, xgboost, catboost)은 기본 설치에 없으므로 별도 명시 필요.**

시스템 패키지: `git`, `build-essential`, `python3.11-dev`, `libffi-dev`, `libblas3`,
`liblapack3`, `libomp-dev`, `unzip`, `p7zip-full`, `gfortran`, `libatlas-base-dev`,
`default-jre-headless`, `cmake`, `pkg-config`, `ninja-build`, `libgl1`, `libglib2.0-0`

---

## 6. 주최 측 베이스라인

```python
Pipeline([
    ("pre", ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         ["top_bottom", "game_type", "base_state"]),
        ("num", SimpleImputer(strategy="median"), <나머지 44개>),
    ])),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                   min_samples_leaf=200, n_jobs=-1, random_state=42)),
])
```

- 피처 목록은 **`test.csv`의 컬럼이 정한다** (`row_id` 제외 47개)
- 학습 노트북의 검증 분할: 2019–2023 학습 → 2024 검증
- 최종 저장 모델은 2019–2024 **전체 재학습본** (`model/rf.pkl`, 3.96MB)
- 우리 재현 결과: 2024 검증 Brier 0.248769, **Score 415.57** (공식 출력 및 `rf.pkl` 예측과 완전 일치)
- 리더보드(2025) 점수: **549.51**
