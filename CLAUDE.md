# CLAUDE.md

LG Aimers 9기 — 투구 제구 성공 확률 예측 (Dacon 코드 제출 대회).
투구 하나의 제구 성공 **확률**을 예측한다. 평가는 Brier Skill Score.

이 파일은 판단 기준만 담는다. 사실은 아래에 있다.

## ⚠️ 현재 파이프라인 = `wonwoo_repro/` (2026-08-23부로 전환, 사용자 지시)

s01(wonwoo 순수 재현) 실제 LB **1078.84**로 원본과 일치 확인, s02(경험 티어
슬로프 사후보정 추가) 실제 LB **1080.72**(Δ**+1.883**, 확정 이득)까지 나온
뒤 우리 자체 51피처 LightGBM+CatBoost 파이프라인(`src/`)은 **개발 중단·보관**
으로 전환했다. 이후 모든 신규 실험·제출은 `wonwoo_repro/`를 기반으로 진행한다.

- 피처 정의 단일 진실 소스: `wonwoo_repro/src/features.py`
  (`fit_artifacts`/`build_features` — 78개 모델 입력 피처, 목록·구조는 `model.md`
  "wonwoo 2-fold 프로토콜 완전 전환" 절 이후 기록 참고)
- 사후 로짓 보정 축(모델 피처 아님): `logit_adjust()` — ctx(hand/count4) + asof
  시즌진행분(gap_success/gap_bsuccess) + gt_F 델타 + **티어 슬로프(s02, 우리 추가)**
- 추론 코드: `wonwoo_repro/src/predict.py` (제출 zip 루트에 그대로 복사되는
  배포 코드 — `config`/`data`/`validation` 등 학습 전용 모듈 의존 없음)
- 제출 빌드: `wonwoo_repro/src/build_submit.py` — `next_zip_path()`가
  `meta.json`(=`config.py` 하이퍼파라미터) 기준 중복 가드를 건다. `predict.py`만
  바뀐 경우(우리 티어 보정처럼) 가드를 못 잡으므로 `suffix=` 수동 오버라이드 필요
- 검증: `wonwoo_repro/src/validation.py`(`bss`/`shift_to_prior`/`decompose` 등
  wonwoo 원본) — 이게 유일한 검증 기준. `src/verify_pipeline_calib.py`(우리가
  추가, 티어 캘리브레이션 진단 포함)가 이 위에서 돈다
- `wonwoo_repro/`는 **git-untracked 로컬 샌드박스**다. 정식 버전관리 편입 여부는
  아직 사용자 결정 대기 — 그 전까지 존재를 전제하지 말고 경로로 참조할 것
- `wonwoo_repro/SUBMISSIONS.md`가 실제 제출 이력 표(파일명/Trackman/prior/피처/
  홀드아웃/LB) — 새 제출마다 채운다

우리 자체 `src/`(51피처) 파이프라인·`experiments/exp01~66`·`results.csv`는
**유효한 과거 기록으로 보관**하고 재실행하지 않는다. 아래 "검증 프로토콜"·
"폐기 목록"·"Known Pitfalls"의 서술 다수가 그 파이프라인 기준으로 쓰였지만,
방법론적 교훈(축소 강도 통일·클리핑 점수 금지·fit fold≠eval fold 등)은
wonwoo_repro 작업에도 그대로 적용된다 — 파이프라인은 바뀌었어도 판단 기준은
안 바뀐다.

| 문서 | 언제 |
|---|---|
| `EDA_정리.md` | **작업 시작 전 항상.** 확정된 분석 결과 (원본 데이터는 동일하므로 여전히 유효) |
| `competition.md` | 대회 규칙·서버 환경 |
| `model.md` | 실험사 전체 기록 — 현재 진행 중인 라인은 "wonwoo 2-fold 프로토콜 완전 전환" 절 이후 |
| `wonwoo_repro/SUBMISSIONS.md` | 실제 제출 직전·직후 |

---

## 작업 원칙

**측정하지 않은 개선은 개선이 아니다.** BSS 0.005 스케일 싸움이다.
주장하려면 폴드·Δ·CI 세 가지가 있어야 한다.

**모호하면 물어라.** 조용히 한 해석을 골라 진행하지 마라.
내 분석의 오류는 지적하라 — 이미 측정 버그가 세 번 나왔다.

**한 번에 하나만 바꾼다.** 피처를 추가하면서 전처리를 "개선"하지 마라.
검증 하니스(`src/folds.py`, `src/metrics.py`)는 실험 중 건드리지 않는다.

**성공 기준을 먼저 말한다.** "피처 추가해줘"가 아니라
"이 피처의 `pts_res`가 517.5를 넘는지"로 바꿔서 시작한다.

---

## 불변식 — 깨지면 결과 전체가 무효

**strictly-prior 누적 피처는 `cumsum() - 자기값`.** 빠뜨리면 에러 없이 점수만 좋아진다.

```python
# GOOD
cum_n = df.groupby("pitcher_id")["is_R"].cumsum() - df["is_R"]
cum_y = df.groupby("pitcher_id")["y_R"].cumsum() - df["y_R"]
rate  = np.where(cum_n > 0, cum_y / cum_n, np.nan)

# BAD — 현재 투구가 자기 이력에 포함됨
rate = df.groupby("pitcher_id")["y"].expanding().mean()
```

새 누적 피처마다 직접 계산과 대조하는 단위 테스트를 같이 쓴다.
`artifacts/decoded_pitch_outcomes.parquet`는 현재 투구의 **사후 정보**다.
strictly-prior 집계 재료로만 쓰고, 현재 행에 조인하면 라벨 누출이다.

**피처 정의는 `wonwoo_repro/src/features.py` 한 곳에만.** `predict.py`가 이걸
import 한다(재구현 아님) — 학습·검증·추론 세 경로가 `build_features`/
`logit_adjust` 같은 함수를 봐야 §7.36류 버그(축 하나가 세 경로 중 한 곳에서만
빠지는 것)가 안 생긴다. 새 사후보정 축을 추가할 땐 `CTX_CODERS`(또는 같은 구조의
`asof`/`mix` 리스트) 한 곳만 고친다 — 티어 슬로프처럼 `predict.py`에 직접 넣는
방식은 예외(모델 재학습 없는 순수 로짓 후처리라 별도 아티팩트 구조가 필요 없었음).
(우리 자체 `src/features.py`는 archived — 더 이상 단일 진실 소스가 아니다.)

**anchor는 실험마다 바꾸지 않는다.** wonwoo_repro는 `submit/model/`에 학습
아티팩트를 한 번 굳히고 재사용한다(s01→s02 전환도 재학습 없이 `predict.py`
패치만으로 이뤄졌다) — 비교할 땐 이 동결된 아티팩트 기준으로 Δ를 본다.

**점수는 리더보드 환산으로만 보고.** 원시 Brier는 폴드마다 $r$이 달라 비교 불가.

```python
Score   = max(0, 1e5 * (1 - BS / (r*(1-r))))            # r = 검증 집합 실제 base rate
pts_res = 1e5 * (1 - mean((p - bias - y)**2) / den)     # 오라클 절편 = 해상도만
# ΔBS 1e-4 ≈ 40점
```

---

## Known Pitfalls

전부 실제로 당한 것들이다.

| DO NOT | 증거 |
|---|---|
| 배포된 `rf.pkl`을 anchor로 쓰기 | 전 시즌을 봤고 폴드마다 오염 정도가 다름. 폴드별 재학습 필수 |
| 절대 점수로 모델 비교 | 같은 예측 절반씩 = 415.57 vs 376.77. 잡음 바닥이 수십 점 |
| `max(0,...)` 클리핑된 점수로 피처 순위 | base rate 불일치만으로 −270점. 해상도 낮은 피처가 전부 0으로 잘림 |
| 축소 강도 다른 두 방법 비교 | 계층 $k$=50 vs 전역 $k$=200. 바뀌는 변수는 하나여야 함 |
| 결측을 median 대치 | 투수 lookup `miss_pct` 20.69%. NaN 유지 + `asof_*` fallback 2단 |
| F를 섞어서 평가 | fold A 2077점 중 1820이 F 아티팩트. 평가는 `game_type=="R"` 전용 |
| test 참조 상수 만들기 | 절편·base rate·캘리브레이션 전부 학습 시점 확정 후 `model/` 저장 |
| RF 튜닝 | 47피처 RF(420.48) ≈ 단일 피처 lookup(420.47). 탐색은 LightGBM |

---

## 검증 프로토콜

**2026-08-21부로 wonwoo 2-fold(확장 윈도우) 방식으로 완전 대체(사용자 지시),
2026-08-23부로 wonwoo_repro 파이프라인 자체에서 직접 검증(사용자 지시).**
옛 A/B/C/D(롤링 3시즌 창)는 폐기 — `experiments/exp04~64`는 그 정의로 기록된
과거 결과이므로 재실행하지 말고 당시 수치를 그대로 읽는다 (`src/folds.py` 주석 참고).
`src/folds.py`를 2-fold로 바꾼 뒤 우리 자체 파이프라인 위에서 재현한 exp65/66도
같은 이유로 이제 과거 기록이다 — 2-fold 정의 자체(폴드 구간·지표)는 아래
그대로 유효하지만, **적용 대상은 `wonwoo_repro/src/validation.py` +
`src/verify_pipeline_calib.py`이지 우리 `src/folds.py`가 아니다.**

| fold | train | valid | 용도 |
|---|---|---|---|
| 2023 | 2019–2022 | 2023 | 모델 선택 |
| 2024 | 2019–2023 | 2024 | **드리프트 대응 주근거 · 최종 확인** |

확장(forward-chaining) 윈도우 — 매 fold가 이전 fold의 학습 구간을 그대로 포함한다.

- 학습에서 `game_type=="F" & season<=2022` 105,308행은 **항상 제외** (튜닝 대상 아님)
- 평가는 R 전용. 클러스터 부트스트랩은 **투수 단위** (경기 ID가 데이터에 없음)
- 지표: `BSS(served)`(prior-shift 포함 실배포, 우리 `score()`와 동형) +
  `BSS(oracle)`(평가셋 실제 평균으로 레벨 맞춘 뒤 순수 해상도, 우리 `score_pair()`의
  `pts_res`와 같은 목적 — wonwoo는 logit 공간 상수이동, 우리는 확률 공간 상수차감이라
  완전 동일값은 아님. 필요시 통일 검토, 지금은 두 구현 그대로 병존)
- 판정: ① seed 노이즈 게이트(2σ, exp65 실측: 2023=14.5·2024=6.3pts, LightGBM 단일
  기준, exp04 config 5seed) → ② 2023/2024 두 fold 부호 일치 → ③ 크기비 ≤3배 →
  ④ 클러스터 부트스트랩 CI(우리 도구, 최종 관문 — wonwoo에는 없는 단계)
- 사후 보정(β 등) 검증 시 **fit fold ≠ eval fold 분리** (wonwoo §7.83) — 2023에서
  fit→2024에서 평가, 반대로도 — 같은 fold에서 fit+평가하면 항상 유리하게 나온다
- 층화 축: `asof_pitcher_n` 구간 / `game_type` / `pitcher_team_id==13` / 볼카운트 / 월

**실험 하나 = `results.csv` 한 줄.** 스냅샷("depth=8 → 452점")은 쓰지 마라.

```
label, fold, delta_pts, ci_lo, ci_hi, slope, n_va, seed, git_sha
```

---

## 코드에 직접 영향을 주는 사실

전체는 `EDA_정리.md`.

- `train` = 2019–2024, `test` = **2025 시즌**
- `row_id` 완전 연대순. `asof_pitcher_n`은 gap 없는 0-based 전역 누적 (시즌 리셋 없음).
  test의 `asof_n`은 train 끝에서 이어진다
- `pitcher_hand`/`batter_hand`: **2 = 우, 1 = 좌**
- `base_state` 실제 값은 `___`, `1__`, `_2_` … (`*2*`로 보이면 마크다운 렌더링)
- **2022 이전 F는 라벨 생성 과정이 다르다** — 같은 투수 F−R 격차가 +0.244 → +0.011로 붕괴
- R base rate `.5495 → .4897` (2019→2024), $r_{2025}$ 추정 **0.4745~0.4897**.
  `season`은 반드시 피처로
- 최고 피처는 `asof_success_R_platoon_blend`(R 전용 재구성 + strictly-prior 좌우 매치업 분할,
  노트북 05) — 4폴드 평균 600.7. `asof_success_R2`(517.5) 대비 +85~+97, 주최 제공(434.6) 대비 훨씬 큼.
  콜드스타트 무시 가능(2024 R `platoon_n` 결측 0.05%). `pitcher × batter_hand` 그룹 lookup은
  `pitcher only` 대비 +79.4였지만 이건 별개 방식(strictly-prior 연속 누적).
  이 항목은 **archived된 우리 자체 `src/` 파이프라인** 기준 기록 — wonwoo_repro는
  이와 별개로 `asof_pitcher_success_rate_sm`(경험적 베이즈 스무딩, k=200) +
  시즌진행분 gap 보정 축으로 같은 신호를 담는다(구조는 다르지만 목적 동일)
- **이 문제는 사실상 "투수 제구력 + base rate"다.** 상황·타자 정체성·구종 배합은 해상도 ≈ 0

### `artifacts/` (archived — 우리 자체 `src/` 파이프라인 전용, 노트북 02~03 산출물)

`decoded_pitch_outcomes.parquet` · `rebuilt_asof_features.parquet` ·
`pitcher_train_totals.parquet`(추론에 필요) · `anchor_preds.npz`(동결) — 이 넷은
`wonwoo_repro/`가 참조하지 않는다. wonwoo_repro 쪽 학습 산출물은
`wonwoo_repro/submit/model/`(부스터·MLP·`artifacts.joblib`·`meta.json`).

---

## 폐기 목록 — 다시 제안하지 말 것

RF 튜닝 · isotonic/Platt/기울기 보정(held-out −13.4/−2.5) ·
절편(prior-shift) 보정 — nested-proxy(exp06)·추세선 외삽(exp40) 둘 다 fold 부호
큰 폭 불일치로 기각 (+83.2는 RF 앵커 전용, `season` 피처·시즌-동결 asof 피처가
있는 우리 모델엔 안 전이됨. R base rate가 계단형이라 3시즌 창 외삽이 불안정) ·
최근 시즌 지수 가중치(recency weight, exp38~39) — 3시즌 fold 창에서 효과 잡음과
구분 안 됨 (wonwoo 브랜치는 6시즌 창이라 다름) ·
CatBoost `depth` 10 이상(exp37) — 8이 정점, 10은 fold C −42.8로 반전 ·
CatBoost에서 `pitcher_team_id`/`batter_team_id` 제거(exp43~44) — fold A만 +15.7로
보였으나 fold C −87.4로 반전(team_id는 fold C 성능의 핵심 구성요소였음), fold A
원인은 미해결로 남음 · 선형 회귀를 세 번째 앙상블 멤버로 고정 가중치 블렌드(exp45)
— fold A/C는 소폭 이득(+2~+7)이나 fold B가 전 가중치에서 그보다 훨씬 크게(−4.5~
−37.6) 악화, linear 단독 성능이 fold B에서만 붕괴(원인 미상) ·
투수→팀→리그 3단 축소 피처(exp49~50, LB 실측) — 로컬 5seed 평균도 fold C만
음수로 판정 보류였는데 실제 LB에서 챔피언 대비 −4.75로 확정 기각 ·
표본크기(`asof_pitcher_n`) 조건부 앙상블 블렌드(exp52) — 진단 결과 CatBoost vs
LightGBM 우열이 n이 아니라 fold(학습기간) 단위로 갈려 조건부 규칙 설계 전제 자체가
성립 안 함 ·
LightGBM/CatBoost 시드 앙상블(exp54~56, 59) — 개별 모델 단위로는 확실한 분산
축소 효과(전 fold 양수)지만 50/50 블렌드+ctx보정에 통합하면 4가지 조합 전부
CI가 0을 포함해 확정 안 됨(fold C는 CatBoost 시드평균이 원흉, −8.3), 90개 이상
모델로 철저히 검증 후 완전 종결 ·
사후 ctx보정 축 확장 — outs/base_state/top_bottom/inning3(exp57) 전부 무효과,
hand×count4 결합축(exp58)도 데이터 희석으로 소폭 손해 — hand+count4 애디티브가
이 메커니즘의 최종형, 초구(0-0카운트) 축(exp61)도 채널 포화로 기각 ·
게임 경계 복원 기반 원시 피처 — 이닝당 투구수·당경기 매치업 반복횟수(exp62/63,
경계 복원 로직 자체는 검증됨·4872경기) 둘 다 fold A/B 양수·C 음수로 부호 불일치,
Trackman과 같은 "fold마다 관계 방향 반전" 패턴 ·
`far_off`·`overlap` 누적 피처(−19~+27) · 타자 정체성·`li`·WE·`inning`·구종 배합 ·
카운트 조건부 투수 이력(−30.9) · Trackman ID 매핑 · test 기반 importance weighting ·
fold A/B 단독 판단
