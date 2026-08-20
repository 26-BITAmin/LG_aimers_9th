# 재현 절차 — Private Score 재현용 학습 코드 안내

DACON / LG Aimers 9기 Phase 2 「투구 제구 성공 확률 예측」 (대회 ID `236743`)
팀 **끼끼**

이 문서 하나로 최종 제출 아티팩트를 **처음부터 다시 만들 수 있다.**
명령 세 줄이고, 이 저장소의 다른 파일은 읽지 않아도 된다.

| | |
| --- | --- |
| 최종 제출 | `s12_p475_e20m8a40pl15c10.zip` |
| Public LB | **984.7488644607** |
| 재현 소요 | 약 **30분** (아래 하드웨어 기준) |
| 결정성 | 시드 고정. 동일 환경에서 **비트 단위로 동일한 산출물** |

---

## 1. 환경

학습에 실제로 쓴 환경이다. **버전을 정확히 맞춰야 한다** — 직렬화 포맷과
부동소수점 연산이 마이너 버전에서도 달라질 수 있다.

| | |
| --- | --- |
| OS | Windows 11 Home 26200 (x64) |
| Python | **3.13.14** (CPython, MSC v.1944 64bit) |
| CPU | 16 논리 코어 |
| GPU | **쓰지 않는다.** torch는 CPU 빌드로 고정 |
| RAM | 학습 피크 약 12 GB |

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

`requirements-dev.txt`의 전체 목록:

```
lightgbm==4.6.0   numpy==2.2.6      pandas==2.3.3     scipy==1.15.2
joblib==1.4.2     scikit-learn==1.6.1                 pyarrow==19.0.1
torch==2.13.0+cpu   (--index-url https://download.pytorch.org/whl/cpu)
```

> **⚠️ torch는 반드시 `+cpu` 빌드.** CUDA 빌드는 부동소수점 누산 순서와 RNG가
> 달라 같은 시드로도 다른 가중치가 나온다. MLP 8멤버가 배포 블렌드의 40%를
> 차지하므로, 빌드가 다르면 기록된 LB 점수를 재현할 수 없다.
>
> torch는 **학습에만** 쓴다. 학습이 끝나면 가중치를 npz로 내보내고 추론 순전파는
> numpy가 한다(`src/mlp_infer.py`). 그래서 **제출 zip에는 torch가 들어가지 않는다.**

---

## 2. 데이터 배치

대회에서 받은 파일을 그대로 둔다. 가공하지 않는다.

```
data/
├── train.csv                 1,475,092 × 49   (2019~2024)
├── test.csv                          5 × 48   ← 형식 확인용 샘플
└── sample_submission.csv             5 × 2
```

`trackman_history.csv`는 **최종 모델이 쓰지 않는다** (`config.TRACKMAN_MODE = "none"`).
없어도 재현된다. 기각 경위는 `MODEL.md` §7.9 · §7.39.

---

## 3. 실행 — 명령 두 줄

```bash
.venv/Scripts/python.exe -u src/train.py 2>&1 | tee experiments/train.log
```

약 **28분**. 산출물은 `submit/model/`.

| 단계 | 소요 | 산출 |
| --- | ---: | --- |
| 피처 빌드 (80개) | 7s | — |
| LightGBM 4변형 × 5시드 = 20개 | 679s | `lgbm_*.txt` |
| MLP 2변형 × (5+3)시드 = 8개 | 975s | `mlp_*.npz` |
| 보정축 δ fit + prior shift | 3s | `artifacts.joblib`, `meta.json` |

정상 실행 시 로그 끝부분:

```
보정축 hand: 셀 1,571개  δ std=0.01758  β=1.5
보정축 count4: 셀 3,105개  δ std=0.01358  β=1.0
  블렌드: (1-0.4)·lgbm + 0.4·mlp   (로짓 상관 0.950)
보정축 참조행 적중 hand 99.9%  count4 100.0%   로짓 기여 평균 +0.00043
prior shift: ref(2024) pred_mean=0.4887 → target=0.4750
  base=+0.00000  delta=-0.05520  → logit_delta=-0.05520
```

```bash
.venv/Scripts/python.exe -u src/build_submit.py 2>&1 | tee experiments/build.log
```

약 **90초**. 저장소 루트에 `s12_p475_e20m8a40pl15c10.zip` (18.2 MB)이 생긴다.
이 스크립트가 하는 일은 패키징만이 아니다:

1. **리키지 정적 검사** — 추론 경로(`script.py`/`predict.py`/`features.py`/`mlp_infer.py`)의
   AST를 훑어 `groupby` `value_counts` `rolling` `expanding` `cumsum` `fit` `quantile`
   `describe` `corr` `cov` `mode` 호출이 있으면 **패키징을 중단**한다
2. **추론 시간 측정** — 245,789행 더미로 실측 (마지막 실행 **26.9s / 600s 한도**)
3. **파일명 태그 생성** — 설정에서 태그를 만들어 같은 이름의 서로 다른 모델이
   생기지 않게 한다

---

## 4. 결정성

시드는 `config.SEED = 42`, 앙상블 시드는 `SEED_ENSEMBLE`에 고정돼 있다.

**평가 서버를 거쳐 확인된 사실:** 제출 01과 04는 같은 설정을 다시 학습한 것인데
LB 점수가 **933.1039119744로 소수점까지 완전히 일치**했다
(`SUBMISSIONS.md` 「재현성 확인」).

동일 환경에서 재현하면 `submit/model/`의 모든 파일이 비트 단위로 같아야 한다.
빠르게 대조하려면:

```bash
.venv/Scripts/python.exe -u src/rowindep.py
```

`통짜 예측 평균`이 **0.474972**로 나오면 같은 모델이다 (2024년 20,000행 표본 기준).

---

## 5. 검증 스크립트

재현 자체에는 필요 없지만, 심사에 쓸 수 있도록 남겨둔다.

| 명령 | 확인하는 것 | 소요 |
| --- | --- | ---: |
| `src/rowindep.py` | **행 독립성** — 조각/셔플/1행씩 예측이 통짜와 같은가 | 2분 |
| `src/ctxcheck.py` | 보정축이 측정 경로와 같은 값을 내는가 | 3분 |
| `src/verify_pipeline.py` | 학습→추론 왕복 전체를 홀드아웃으로 | 40분 |

`rowindep.py`의 기대 출력 — 셋 다 정확히 `0.000e+00`이어야 한다:

```
① 조각 1000행 × 20개 vs 통짜   max|diff| = 0.000e+00   ✅
② 행 순서 셔플 후 복원 vs 통짜  max|diff| = 0.000e+00   ✅
③ 1행씩 200회 vs 통짜          max|diff| = 0.000e+00   ✅
```

---

## 6. 파일 지도

`src/`에 76개 파일이 있는데 **재현에 필요한 것은 12개**다. 나머지는 실험 기록이며,
`MODEL.md`의 실험 로그가 각각을 참조한다 (지우면 근거가 끊긴다).

### 재현 경로 (12개)

| 파일 | 역할 |
| --- | --- |
| `config.py` | 경로·시드·하이퍼파라미터. **모든 설정의 단일 출처** |
| `data.py` | CSV 로딩, dtype 최적화, recency 가중 |
| `features.py` | 피처 빌더 + 사후 로짓 보정. **학습·추론 공용** |
| `ctxfit.py` | 투수×상황 δ 학습 (학습 전용 — 타깃을 쓴다) |
| `numemb.py` | 수치 임베딩 유틸 (`mlp.py`가 import) |
| `mlp.py` | MLP 학습 (torch) + npz 내보내기 |
| `mlp_infer.py` | npz numpy 순전파. **추론 전용** |
| `validation.py` | Brier/BSS, prior shift 풀이 |
| `trackman_features.py` | `TRACKMAN_MODE="none"`이면 `None` 반환 |
| `train.py` | 학습 엔트리포인트 → `submit/model/` |
| `predict.py` | 추론 로직. `build_submit.py`가 zip에 복사한다 |
| `build_submit.py` | 리키지 검사 + 시간 측정 + 패키징 |

`submit/script.py`가 평가 서버의 엔트리포인트이며 `predict.py`를 호출한다.

### 검증 (5개)
`verify_pipeline.py` `ctxcheck.py` `rowindep.py` `foldcache.py` + 위 재현 경로

### 실험 기록 (60개)
`ablate*.py` 34개 · `eda*.py` 5개 · `ctxsplit*.py` `platoon*.py` `tmconf.py`
`tmresid.py` `lbcalib.py` `decide.py` `foldsurvey.py` `trackman_map.py` 등.
**재현에 불필요하다.** 각각 `MODEL.md` §7의 해당 절이 결과를 기록하고 있다.

---

## 7. 규정 준수

CLAUDE.md §3.2의 리키지 규칙에 대한 증빙이다.

| 규칙 | 이 저장소가 지키는 방법 |
| --- | --- |
| 투구 이후 확정 정보 사용 금지 | 피처는 `asof_*`(운영측 제공, 투구 직전 시점)와 상황 컬럼만 |
| **test.csv 다른 행을 이용한 피처 금지** | ① AST 정적 검사가 집계 호출을 차단 ② `rowindep.py`가 조각·셔플·1행 예측의 동일성을 실측 |
| 외부 데이터 금지 | 대회 제공 `train.csv`만. Trackman도 제공 데이터이나 최종 모델은 미사용 |
| 원격 API 금지 | 전부 로컬. 학습·추론 모두 네트워크 불필요 |
| 2025 Trackman 금지 | Trackman 자체를 안 쓴다 (`TRACKMAN_MODE="none"`) |

**train에서 fit → test에 적용**만 한다 (CLAUDE.md가 명시적으로 허용하는 형태):
스무딩 상수, 범주 LUT, 결측 대체값, 투수×상황 δ 테이블, prior shift 스칼라.
전부 학습 시점에 동결돼 `artifacts.joblib`에 들어가고, 추론은 **행 단위 조회만** 한다.

---

## 8. 모델 개요

자세한 근거는 `MODEL.md`. 한 문단으로:

```
z = 0.6 · mean(LightGBM 20개)  +  0.4 · mean(MLP 8개)        ← 로짓 공간 블렌드
z = z + 1.5·δ(투수, 타자손잡이) + 1.0·δ(투수, 카운트4칸)      ← 사후 로짓 보정
p = sigmoid(z + logit_delta)                                  ← 2025 prior shift
```

- **로짓 공간에서 평균**한다. 확률 평균은 예측을 0.5로 수축시켜 캘리브레이션을 해친다
- **δ는 피처가 아니라 사후 보정항**이다. 투수가 특정 상황에서 자기 평균 대비 얼마나
  벗어나는지를 train에서 재어 얼려둔 것이고, 모델 재학습이 없다
- **prior shift는 스칼라 하나**다. 시즌 드리프트를 레벨로만 다룬다 (EDA §H가
  "드리프트는 레벨이지 구조가 아니다"를 확인했다)
