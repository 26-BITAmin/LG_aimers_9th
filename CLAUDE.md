# CLAUDE.md

LG Aimers 9기 — 투구 제구 성공 확률 예측 (Dacon 코드 제출 대회).
투구 하나의 제구 성공 **확률**을 예측한다. 평가는 Brier Skill Score.

이 파일은 판단 기준만 담는다. 사실은 아래에 있다.

| 문서 | 언제 |
|---|---|
| `EDA_정리.md` | **작업 시작 전 항상.** 확정된 분석 결과 |
| `competition.md` | 대회 규칙·서버 환경 |
| `submit/CHECKLIST.md` | 제출 직전 |
| `results.csv` | 새 실험 설계 전 |

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

**피처 정의는 `src/features.py` 한 곳에만.** `submit/script.py`는 이걸 import 하거나
파일째 복사한다. 재구현하면 평가 서버에서만 터진다.

**anchor(`artifacts/anchor_preds.npz`)는 동결.** 실험마다 바꾸면 Δ의 의미가 달라진다.

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

| fold | train | valid | 용도 |
|---|---|---|---|
| A | 2019–21 | 2022 | 모델 선택 (F 아티팩트 주의) |
| B | 2020–22 | 2023 | 모델 선택 |
| C | 2021–23 | 2024 | **드리프트 대응 주근거** |
| D | 2019–23 | 2024 | **최종 확인 전용 — 재튜닝 금지** |

- 학습에서 `game_type=="F" & season<=2022` 105,308행은 **항상 제외** (튜닝 대상 아님)
- 평가는 R 전용. 클러스터 부트스트랩은 **투수 단위** (경기 ID가 데이터에 없음)
- 판정: ⓪ seed 노이즈 게이트(2σ, exp47 실측: A=9.6·B=12.1·C=15.2pts, LightGBM 단일
  기준) → ① A/B/C 부호 일관성 → ② 점수 환산 크기 → ③ CI (wonwoo `decide.py` 도입)
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
  `pitcher only` 대비 +79.4였지만 이건 별개 방식(strictly-prior 연속 누적)
- **이 문제는 사실상 "투수 제구력 + base rate"다.** 상황·타자 정체성·구종 배합은 해상도 ≈ 0

### `artifacts/` (노트북 02~03 산출물, 재생성 불필요)

`decoded_pitch_outcomes.parquet` · `rebuilt_asof_features.parquet` ·
`pitcher_train_totals.parquet`(추론에 필요) · `anchor_preds.npz`(동결)

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
이 메커니즘의 최종형 ·
`far_off`·`overlap` 누적 피처(−19~+27) · 타자 정체성·`li`·WE·`inning`·구종 배합 ·
카운트 조건부 투수 이력(−30.9) · Trackman ID 매핑 · test 기반 importance weighting ·
fold A/B 단독 판단
