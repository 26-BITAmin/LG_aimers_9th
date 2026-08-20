# CLAUDE.md

이 저장소는 **DACON / LG Aimers 9기 Phase 2 온라인 해커톤 — "투구 제구 성공 확률 예측"** (대회 ID `236743`) 참가용 작업 공간이다.

> 대회 페이지: https://dacon.io/competitions/official/236743/overview/description

---

## 1. 문제 정의

각 행은 **투구 1개의 직전 시점 상태**다. 투구가 던져지기 전에 알 수 있는 정보만으로,
그 투구가 **제구에 성공할 확률**(`control_success` = 1일 확률)을 예측한다.

제구 **실패**로 정의되는 3가지 경우:
1. 스트라이크 존 한가운데로 몰린 공
2. 존에서 크게 벗어난 공
3. 포수 요구와 반대 방향으로 간 공

→ 나머지는 모두 성공. 즉 **이진 분류의 확률 예측(probability estimation)** 문제.

---

## 2. 평가 지표 — Brier Skill Score

```
Brier Score = mean((p_i - y_i)^2)
Score       = max(0, 100000 * (1 - BS_model / BS_reference))
```

`BS_reference`는 "평균 제구율"을 모든 행에 상수로 찍었을 때의 Brier Score
(= `p̄(1-p̄)`, 2025 시즌 평균 제구율 기준이면 대략 0.249).

**이 지표가 작업 방식에 주는 제약 — 반드시 지킬 것:**

- **Proper scoring rule이다.** AUC/accuracy 최적화가 아니라 **확률 캘리브레이션**이 점수에 직결된다.
  순위가 완벽해도 확률이 편향되면 점수가 깎인다. 모든 실험 평가는 **Brier / BSS**로 한다.
  (AUC·LogLoss는 보조 지표로만 기록한다.)
- **점수 스케일이 10만 배로 증폭된다.** Brier 0.0001 개선 = BSS 약 40점. 노이즈와 진짜 개선을
  구분하려면 검증 fold를 늘리고 seed를 고정해야 한다. 단일 홀드아웃 1회 결과로 채택 결정하지 않는다.
- **`clip(p, eps, 1-eps)` 같은 안전장치는 무해하지만**, 확률을 0/1로 밀어붙이는 후처리는
  Brier에서 크게 손해다. 극단값 예측은 확신이 있을 때만.

---

## 3. 데이터

```
data/
  train.csv              1,475,092 x 49   (2019~2024 시즌, control_success 포함)
  test.csv                       5 x 48   ← 형식 확인용 샘플. 실제 평가는 약 245,789행 (2025 시즌)
  sample_submission.csv          5 x 2    ← 마찬가지로 샘플
  trackman_history.csv   1,793,078 x 30   (2019~2024 Trackman 로그, 별도 테이블)
data_description.md                       ← 컬럼 전수 설명. 피처 작업 전 반드시 참조
baseline_submit.zip                       ← 제출 zip 구조 참고용 (RandomForest 베이스라인)
```

### 3.1 반드시 알아야 할 데이터 사실

| 사실 | 함의 |
| --- | --- |
| train = 2019~2024, test = **2025** | 랜덤 KFold는 부적절. **시간 기반 검증**이 기본 |
| 시즌별 양성률: `.565 → .533 → .533 → .529 → .500 → .486` | 뚜렷한 **하향 드리프트**. 2025는 0.486보다 더 낮을 가능성이 높다. Prior shift 보정을 반드시 검토 |
| `game_type`: `R` 1,314,088 / `F` 161,004 | 정규시즌 / 그 외. 분포가 다를 수 있으니 분리 검토 |
| 전체 양성률 0.5238 | 클래스 균형은 거의 맞음 → 리샘플링 불필요 |
| `asof_*` 컬럼 다수가 시즌 초 / 신인에서 **결측** | cold-start 처리(스무딩·fallback) 설계가 성능에 직결 |
| `trackman_history.csv`는 **1:1 조인 테이블이 아님** | `pitcher_trackman_id`는 `pitcher_id`와 다른 ID 체계. 선수 단위 요약 피처로만 쓸 수 있고, ID 매핑 자체가 별도 과제 |

### 3.2 절대 위반 금지 — 리키지 규칙

대회 규정상 아래는 **실격 사유**다. 코드 작성 시 항상 확인한다.

- 현재 투구 **이후에 확정되는 모든 정보** 사용 금지
- 현재 투구의 실제 코스/판정/결과/구종/Trackman 측정값 사용 금지
- **2025년 Trackman 데이터** 사용 금지 (애초에 제공되지 않음)
- **`test.csv`의 다른 행을 이용한 어떤 피처도 금지**
  → 누적 통계, 빈도 인코딩, 분포 통계, target encoding, 행 순서 기반 rolling/expanding, 사후 보정값 전부 해당.
  **각 행은 완전히 독립적으로 예측되어야 한다.**
- 외부 데이터 사용 금지 (대회 제공 데이터만). 사전학습 가중치는 MIT/Apache 2.0 등
  공개·비상업 허용 라이선스에 한해 허용.
- 원격 API(OpenAI, Gemini 등) 호출 금지. 전부 로컬·오프라인 재현 가능해야 함.

**허용되는 것:** 운영측이 미리 계산해준 `asof_*` 컬럼, `train.csv` 전체 통계로 만든
매핑을 `test`에 **적용**하는 것(= train에서 fit, test에서 transform).

> `test` 전체를 `pandas`로 읽어 `groupby`를 돌리는 순간 규정 위반이다.
> 추론 코드에서 `test` 데이터프레임에 대한 `groupby` / `value_counts` / `rolling` /
> `fit` 계열 호출이 있는지 제출 전 반드시 검사한다.

---

## 4. 제출 규격

**코드 제출 방식**이다. CSV만 내는 대회가 아니다.

```
submit.zip
├── model/            ← 학습된 가중치·피처 아티팩트
├── script.py         ← 평가 서버가 자동 실행하는 엔트리포인트
└── requirements.txt  ← pip 설치 가능한 패키지 목록
```

`script.py`의 계약:
- `./data/test.csv`, `./data/sample_submission.csv`를 읽는다
- `./output/submission.csv`에 `row_id,control_success` 2컬럼으로 저장
- `row_id`는 평가 서버 `test.csv`와 **정확히 일치**해야 한다
  (베이스라인처럼 `sample_submission`의 `row_id` 순서에 맞춰 병합하는 패턴을 유지할 것)

**평가 환경 제약:**

| 항목 | 한도 |
| --- | --- |
| 추론 시간 | **10분** (245,789 샘플) |
| 패키지 설치 시간 | 10분 |
| zip 크기 / 압축 해제 후 | 10GB / 32GB |
| 하드웨어 | 6 vCPU, 28GB RAM, L4 GPU (22.4GiB VRAM) |
| 네트워크 | **오프라인** |
| 언어 | Python only (`.py` / `.ipynb`, UTF-8) |
| 일일 제출 | **5회** |

> 6 vCPU / 10분은 생각보다 빡빡하다. 대형 앙상블·행 단위 파이썬 루프는 위험.
> 벡터화된 전처리 + GBDT `predict` 정도가 안전 구간이다. 제출 전 로컬에서
> 245,789행 더미로 추론 시간을 측정한다.

`requirements.txt` 버전은 **학습에 쓴 버전과 정확히 일치**시킨다. pickle은 라이브러리
마이너 버전만 달라도 깨진다. (베이스라인 예시: `scikit-learn==1.8.0`, `joblib==1.5.3`, `pandas==2.3.3`)

---

## 5. 일정

| 날짜 | 내용 |
| --- | --- |
| 2026-08-05 | 대회 시작 |
| 2026-08-26 | 팀 병합 마감 |
| **2026-09-01** | **리더보드 제출 마감** |
| 2026-09-02 | 대회 종료 |
| 2026-09-07 | 코드/PPT 제출 마감 |
| 2026-09-11 | 코드 검증 마감 |
| 2026-09-14 | Phase 3 오프라인 해커톤 진출자(약 100명) 발표 |

> Private Score 재현용 **학습 코드**와 개발 환경 문서(OS·라이브러리 버전)도 제출해야 한다.
> → 실험을 "나중에 정리"하지 말고 처음부터 재현 가능한 스크립트로 남긴다.

---

## 6. 저장소 구조 (목표)

```
.
├── CLAUDE.md              ← 이 파일. 대회 규칙·제약·컨벤션
├── MODEL.md               ← 모델링 전략과 실험 로그
├── data/                  ← 원본 CSV (git에 올리지 않음)
├── src/
│   ├── config.py          ← 경로·시드·상수
│   ├── data.py            ← 로딩, dtype 최적화
│   ├── features.py        ← 학습/추론 공용 피처 빌더 (단일 진실 소스)
│   ├── validation.py      ← 시간 기반 split, Brier/BSS 계산
│   ├── train.py           ← 학습 → model/ 아티팩트 저장
│   └── predict.py         ← script.py가 호출하는 추론 로직
├── experiments/           ← 실험별 설정·결과 (재현용)
├── submit/                ← 제출 zip 빌드 산출물
│   ├── model/
│   ├── script.py
│   └── requirements.txt
└── notebooks/             ← EDA 전용. 모델링 로직은 여기 두지 않는다
```

---

## 7. 작업 컨벤션

- **피처 로직은 `src/features.py` 한 곳에만 존재한다.** 학습 코드와 `script.py`가 같은 함수를
  호출해야 train/serve skew가 안 난다. 베이스라인의 `build_features(df)` 패턴을 유지한다.
- **모든 실험은 `MODEL.md` 실험 로그에 한 줄 추가한다.** CV Brier, CV BSS, LB 점수, 변경점.
  기록 없는 실험은 없었던 것으로 친다.
- **시드는 고정한다** (`SEED = 42`). 결과 비교의 전제다.
- 데이터 파일과 대용량 아티팩트는 커밋하지 않는다. (`.gitignore`: `data/`, `*.csv`, `*.pkl`, `*.zip`)
- 커밋/푸시는 사용자가 요청할 때만 한다.

### 실행 환경 (로컬)

- Python 3.13.14 (Microsoft Store 배포판, `C:\Users\...\WindowsApps\...`)
- 설치됨: `numpy 2.5.1`, `pandas 3.0.5`
- **미설치**: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `torch`
- ⚠️ 로컬 `pandas 3.0.5`는 제출 예시의 `2.3.3`과 다르다. 가상환경을 만들어
  제출용 버전으로 고정하고, 그 환경에서 학습·직렬화까지 수행할 것.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

### 셸 주의사항

Windows / PowerShell 환경이다. `Expand-Archive`, `Get-Content -TotalCount N` 등을 쓴다.
(`head`, `unzip`은 없다. Bash 툴을 쓰면 Git Bash라 POSIX 문법이 되지만 혼용하지 말 것.)

---

## 8. 모델링 전략

→ **[MODEL.md](MODEL.md)** 참조.
