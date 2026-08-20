# PIPELINE.md — 배포 파이프라인의 입력·보정 전수 목록

> 현재 배포 = `s24_nocal62a8.zip`, LB **1067.55** (2026-08-19 기준 최고).
> "무엇이 들어가고, 무엇이 로짓에 더해지는가"만 다룬다. 왜 그렇게 됐는지는 MODEL.md.
> 진실 소스는 코드다: 피처 [`src/features.py`](src/features.py) ·
> 보정 [`src/predict.py`](src/predict.py) + `logit_adjust` · 상수 [`src/config.py`](src/config.py).
> 이 문서와 코드가 어긋나면 **코드가 맞다.**

## 0. 한눈에

```
행 1개 (test.csv)
  → build_features (동결 아티팩트)                      → 피처 78개
  → z_lgb = mean(LGBM 20개 raw score)                   (4변형 × 5시드)
  → z_mlp = Σ 0.125 · MLP_i                             (emb4 5 + noid 3, npz)
  → z = 0.6·z_lgb + 0.4·z_mlp                           (MLP_ALPHA = 0.40)
  → z += δ_ctx(hand) + δ_ctx(count4) + δ_asof(gap 2축)   ← 사후 로짓 보정
  → z += logit_delta (−0.0378, prior shift → 2025 평균 0.475)
  → p = clip(σ(z), 1e-6, 1−1e-6)
```

## 1. 입력 피처 78개 (`build_features` 출력 + ID 2개)

LGBM은 78개 전부를 평면으로 받는다. MLP는 §3처럼 두 채널로 나눠 받는다.

### 경기 상황 — 원본 통과 (17)

`inning` `balls_before` `strikes_before`
`outs_before` `run_top_before` `run_bot_before` `run_total_before`
`score_diff_home` `score_diff_pitcher_team` `runner_on_1b` `runner_on_2b`
`runner_on_3b` `num_runners_on` `home_win_expectancy` `li`
`pitcher_hand` `batter_hand`

### 카운트 파생 (6)

`count_state`(= balls×3+strikes, 0..11) `count_diff` `is_two_strikes`
`is_three_balls` `is_full_count` `is_first_pitch`

### 좌우 매치업 (2)

`hand_same` `hand_matchup`(= ph×4+bh)

### 압박 지표 파생 (4)

`we_margin`(|승리기대−50|) `is_pitcher_home` `abs_score_diff` `is_late_close`

### asof 표본 수 (3)

`asof_pitcher_n` `asof_batter_n` `asof_pitcher_pitchmix_n`

### 결측 플래그 (3)

`miss_pitcher_asof` `miss_batter_asof` `miss_prev_game`
(cold-start 여부 자체가 신호)

### asof 누적률 — 원본 + `_sm` 스무딩 쌍 (10 × 2 = 20)

원본은 NaN 유지(LGBM이 처리), `_sm`은 경험적 베이즈
`(n·rate + k·prior)/(n+k)`, k=200. 대상 10개:

투수(n=asof_pitcher_n): `success` `middle` `reverse` `ball` `strike`
타자(n=asof_batter_n): `success` `middle`
구종믹스(n=pitchmix_n): `fastball` `breaking` `offspeed`

⚠️ 이 중 `pitcher_reverse/success`·`batter_success` 3개는 스무딩 **전에**
pre-2023 F 오염 보정(`contam`, §7.49)이 더해진 값이다 — 파생 전부가
보정된 값을 본다.

### 직전 경기 성적 (6)

`asof_pitcher_prev{1,3,5}_game_{success,middle}_rate`

### 최근 폼 차분 (5)

`form_p1_minus_career` `form_p3_minus_career` `form_p1_minus_p5`
`form_p3_minus_p5` `form_mid_p1_minus_career`
(차분은 트리가 스스로 못 만들므로 명시적으로 넣는다)

### 매치업 엣지 (2)

`matchup_success_edge` `matchup_middle_edge` (투수 − 타자, 스무딩값 기준)

### 실패 유형 구성 (2)

`fail_middle_share`(몰림/(몰림+반대)) `pitcher_ball_minus_strike`

### 구종 레퍼토리 (2)

`pitchmix_entropy` `pitchmix_nonfastball` (3그룹 재정규화 믹스에서)

### 범주형 — 정수 코드, 미지값 −1 (6)

`base_state` `top_bottom` `pitcher_team_id` `batter_team_id`
`pitcher_id` `batter_id` (ID 2개는 `USE_PLAYER_IDS=True`로 원값 통과 —
연대순 정수라 데뷔 시점 프록시를 겸한다)

### 들어가지 **않는** 것 (자주 헷갈리는 것만)

| 컬럼 | 이유 |
| --- | --- |
| `game_type` | `DROP_FEATURES`. 2022→2023 정의 변경으로 독 (§4, 단독 +172) |
| `game_month` `game_dayofweek` | `DROP_FEATURES` (s24, §7.78). 월 패턴은 해마다 달라 전이 안 됨, 요일은 무기전. 블렌드 홀드아웃 +9.4 → LB +0.40 |
| `season` | 트리는 외삽 불가 → 2025를 2024로 취급. 애초에 안 내보냄 |
| Trackman 계열 | `TRACKMAN_MODE="none"`. 홀드아웃 +13.3 → LB −13.6 (§7.9) |

## 2. 학습 때만 개입하는 것 (추론 경로에 없음)

| 장치 | 내용 | 코드 |
| --- | --- | --- |
| soft label F | 2019–22 F & y=1 행 71,199개를 (1, w(1−h)) + (0, w·h)로 복제. h = .09/.11/.24/.30 (시즌 상수, train에서 유도) | `soften.py` §7.70 |
| recency 가중 | w = 0.25^(2024−season). 2019~22 질량 합 6% | `data.recency_weight` |
| MLP 보조 헤드 | [몰림, 반대] 2축, λ=1.0. 라벨은 asof 증분에서 복원 (`modelabel`). export 때 버림 | `mlp.py` §7.41 |

## 3. 모델별 소비 방식

### LightGBM ×20 (base/wide/extra/lowff × 5시드, 320라운드)

80개 평면. NaN 자체 처리. 원본/`_sm` 쌍, ID 원값 모두 분할 후보.

### MLP ×8 (emb4 ×5 + noid ×3, 배포는 npz numpy 순전파)

| 채널 | 컬럼 | 처리 |
| --- | --- | --- |
| 임베딩 7 | `pitcher_id` `batter_id` `pitcher_team_id` `batter_team_id` `base_state` `hand_matchup` `count_state` | 각 4차원 테이블 조회. 미지값 → 0번 행 |
| 수치 71 | 나머지 전부 (`top_bottom` 포함) | NaN → `_sm`/prior 채움 → (x−μ)/σ |

noid 변형은 임베딩에서 `pitcher_id`·`batter_id`를 빼며(5블록),
이때 두 ID는 수치 채널에도 **안 들어간다** — 개인 식별 정보 완전 차단이
변형의 존재 이유(오차 탈상관).

## 4. 사후 로짓 보정 (블렌드 뒤, prior shift 앞 — 순서 고정)

전부 **train에서 얼린 상수/표의 행 단위 조회**다. 모델 입력이 아니다.

### 켜져 있는 축 ✅

| 축 | 내용 | 배포값 | 근거 |
| --- | --- | --- | --- |
| ctx `hand` | 투수×상대손 δ 표 (1,571셀), K=200, hl=1.0 | β=1.875 | §7.36·7.61 |
| ctx `count4` | 투수×카운트4그룹 δ 표 (3,105셀) | β=1.25 | §7.43 |
| asof `gap_success` | 투수의 올해−커리어 성공률 (시즌 시작 상수를 빼 행 단위 복원), 수축 S=35 | β=0.03 | §7.55 (LB +26.14) |
| asof `gap_bsuccess` | 타자판 | β=0.03 | §7.57 |
| prior shift | `logit_delta = −0.0378` 스칼라. 예측 평균 → 0.475 | 학습 시 고정 | §3 |

### 꺼져 있는 축 ❌ (config에 흔적이 남아 헷갈리기 쉬움)

| 축 | 상태 | 왜 |
| --- | --- | --- |
| `gt` δ_F (행 단위 F 레벨) | **`GT_F_BETA=0.0` → 유효 δ=0.** config의 `GT_F_DELTA=−0.0711`은 측정값 기록일 뿐 | s22 프로브 LB −2.81 기각 (§7.69) |
| `mix` r3_ent (레퍼토리 균형) | `MIX_AXES=[]` | LB 기각 후 철회 (§7.58) |

## 5. 아티팩트 대응표 (`submit/model/`)

| 파일 | 내용 |
| --- | --- |
| `lgbm_{변형}_{시드}.txt` ×20 | LightGBM 부스터 (model_str) |
| `mlp_{emb4,noid}_{시드}.npz` ×8 | MLP 가중치 (BN은 Linear에 접힘) + `ecols` |
| `artifacts.joblib` | priors·smooth_k·clip·cat_maps·contam·ctx·asof·gt·mix + `logit_delta` + mlp 인코더/가중 |
| `meta.json` | 설정 스냅샷 (n_features=78 등) — 사람 확인용 |
