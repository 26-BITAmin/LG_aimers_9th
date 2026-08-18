# R / F 별 선수 통계 테이블 (2026-08-17)

원본: `data/train.csv` (1,475,092행). 생성 스크립트: `build_stats.py` → `make_tables.py`.

## 파일

| 파일 | 행 | 단위 |
| --- | ---: | --- |
| `pitcher_season.csv` | 3,127 | 투수 × `game_type` × `season` |
| `pitcher_career.csv` | 1,245 | 투수 × `game_type` (통산) |
| `batter_season.csv` | 3,347 | 타자 × `game_type` × `season` |
| `batter_career.csv` | 1,305 | 타자 × `game_type` (통산) |
| `league_season_summary.csv` | 12 | 리그 전체 × `game_type` × `season` |

`game_type`: `R` = 정규시즌(1,314,088행) / `F` = 그 외(161,004행). **리그 구분이 아니다.**

---

## 두 가지를 복원해서 만들었다

`train.csv`에는 경기 ID도, 투구별 결과 라벨도 없다. 둘 다 파생했다.

### 1. `game_id` — 경기 복원

행이 시간순으로 정렬돼 있다(검증: `asof_pitcher_n`이 투수별로 정확히 +1씩 증가, 시즌 비감소).
`(season, game_month, game_dayofweek, game_type, 팀쌍)`의 연속 블록으로 자르고,
같은 키 안에서 이닝 또는 누적득점이 **감소하면 다른 경기**로 분할했다(더블헤더·같은 요일 재대결).

**검증 — 야구 상식과 대조:**

| 항목 | 복원값 | 실제 |
| --- | ---: | --- |
| R 경기 수 / 시즌 | 720, 720, 721, 720, 720, 721 | KBO 144경기 × 10팀 ÷ 2 = **720** |
| 블록 내 이닝 단조 | **100.00%** | |
| 블록 내 누적득점 단조 | **100.00%** | |
| 경기당 투구 수 중앙값 | **300** | 통상 280~310 |
| 경기당 타석 수 | 76~81 | 통상 76~80 |
| 타석당 투구 수 | **3.87** | KBO/MLB 3.8~3.9 |

`pa_id`(타석)도 같은 방식으로 복원 — `(game_id, pitcher_id, batter_id)` 연속 런. 총 381,377타석.

### 2. 투구 단위 결과 라벨 — `asof` 역산

`asof_pitcher_n`이 리셋 없는 카운터라 `cnt = asof_pitcher_n × asof_pitcher_*_rate`의
**다음 행 증분이 그 투구의 라벨**이다.

**검증: 역산한 `success` 라벨이 `control_success`와 1,474,300행 전부 일치 (100.0000%).**
같은 기전으로 `reverse` `middle` `ball` `strike`(모두 `asof_pitcher_n` 기준)와
`fastball` `breaking` `offspeed`(`asof_pitcher_pitchmix_n` 기준)를 복원했다.

- 투수별 마지막 1행(792행)은 다음 행이 없어 복원 불가 → 비율 계산에서 제외
- 통산 성공률 역산값 vs `asof` 마지막 값 최대편차 **0.00285** (= 마지막 1구 차이, 예상대로)

**라벨 구성 (중요):**

- `fastball + breaking + offspeed = 1` — **100% 완전 분할**
- `success + reverse + middle` — 합=1이 83.4%, 0이 13.2%, 2가 3.4%
  → **네 번째 실패 유형이 따로 있고**(설명서 ②"존에서 크게 벗어난 공"), reverse·middle은 공존 가능
- `ball + strike` — 합=1이 81.3%. 나머지 18.7%는 인플레이로 보인다(추론)

---

## 컬럼 사전

### 공통

| 컬럼 | 뜻 |
| --- | --- |
| `game_type` | `R` 정규시즌 / `F` 그 외 |
| `season` | 시즌 (season 파일만) |
| `role` | `pitcher` / `batter` |
| `player_id` | `pitcher_id` 또는 `batter_id` |
| `games` | 출장(등판) 경기 수 — 복원한 `game_id` 기준 |
| `pitches` | 투구 수 (타자는 상대한 투구 수) |
| `PA` | 타석 수 — 복원한 `pa_id` 기준 |
| `seasons` | 등장 시즌 수 (career 파일만 의미 있음) |
| `teams` | 소속 팀 ID 종류 수 |
| `li_mean` | 평균 상황 중요도 |
| `inning_mean` | 평균 이닝 |
| `pitch_per_game_mean/med/max` | 경기당 투구 수 |

### 투수 전용

| 컬럼 | 뜻 |
| --- | --- |
| `start_rate` | 등판 경기 중 **1회에 등판을 시작한 비율** = 선발 프록시 |
| `entry_inning_med` | 등판 시작 이닝 중앙값 |
| `batter_hand_mean` | 상대 타자 손잡이 코드 평균 (플래툰 노출도) |

`start_rate`는 실측으로 타당성이 확인된다 (R, 통산 200투구 이상):

| `start_rate` | 투수 | 경기당 투구(중앙) | 진입 이닝(중앙) | `li` |
| --- | ---: | ---: | ---: | ---: |
| 0.0–0.1 | 236 | 17.7 | 7 | 0.77 |
| 0.9–1.0 | 112 | **92.8** | **1** | 0.98 |

### 결과 비율 (투구 단위 역산)

`success_rate` `reverse_rate` `middle_rate` `ball_rate` `strike_rate`
`fastball_rate` `breaking_rate` `offspeed_rate`

타자 파일에는 `success_rate` `middle_rate`만 있다 (`asof_batter_*`에 나머지가 없다).

### 상황 비율

`first_pitch_rate`(초구) `two_strike_rate`(2스트라이크) `three_ball_rate`(3볼)
`on_base_rate`(주자 있음) `risp_rate`(득점권)

---

## ⚠️ 2019–2022 F 행의 라벨은 오염돼 있다

역산 라벨은 운영측이 계산한 값을 그대로 복원한 것이지 정답이 아니다.
2019–22 F 구간은 reverse(포수 요구 반대 방향) 검출이 실패해 **실패가 성공으로 기록**됐다.

리그 수준 `reverse_rate`:

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **F** | .1249 | .1512 | **.0917** | **.0482** | .2997 | .2879 |
| **R** | .1884 | .2153 | .2623 | .2586 | .2507 | .2427 |

2022 F의 reverse가 **0.048**까지 내려갔다가 2023에 **0.300**으로 복귀한다.

**이 테이블이 기존 findings를 한 곳 정정한다.** 기존 문서는
"reverse 결손 ~0.11은 F 성공률 격차 0.24의 **일부만** 설명한다"고 적었다 —
그 0.11은 `asof_*`가 R·F를 섞어 누적한 값이라 **하한**이었다. 투구 단위로 보면:

| 2022 F → 2023 F | Δ |
| --- | ---: |
| Δ`success` | **−0.2357** |
| Δ`reverse` | **+0.2515** |
| Δ`middle` | −0.0132 |
| **합** | **+0.0026** (성공률 격차의 1.1%) |

**성공률 급락 전량이 reverse 증가로 상쇄된다.** 시즌별 결손 vs 성공률 초과분도 대응한다:

| 시즌 | reverse 결손 | success 초과분 |
| ---: | ---: | ---: |
| 2019 | +0.169 | +0.223 |
| 2020 | +0.143 | +0.122 |
| 2021 | +0.202 | +0.238 |
| 2022 | +0.246 | +0.243 |

→ **F 오염은 reverse 단일 채널이고, 크기가 정량화됐다.** 필요하면 2019–22 F의 성공 라벨을
정상구간 reverse율(0.294)로 재추정해 부분 복원하는 것도 원리상 가능하다(다만 어느 개별 투구가
reverse였는지는 알 수 없으므로 행 단위 복원이 아니라 가중치·prior 조정에 한정된다).

---

## 재현

```bash
python build_stats.py    # game_id · pa_id · 투구 라벨 복원 → X.pkl
python make_tables.py    # 5개 CSV 생성
```
인코딩 `utf-8-sig`(Excel 바로 열림). 정렬·필터 없이 원시 집계값만 들어 있다.

---

## 알려진 결측

`batter_career.csv` 1행 / `batter_season.csv` 2행에서 `success_rate`·`middle_rate`가 비어 있다.
해당 타자가 통산 1투구만 상대해 **다음 행이 없어 역산이 불가능**한 경우다
(`player_id=23960`, `game_type=F`, 1경기·1투구·1타석). 투수 쪽은 결측 0.

## 검증 로그

```
투구수 합계        1,475,092 = train 원본        OK
R                 1,314,088 = 원본              OK
F                   161,004 = 원본              OK
시즌 × game_type   전 셀 일치                    OK
표본 투수 5명       success_rate·games 직접 재계산 일치  OK
success 라벨       control_success와 1,474,300행 100.0000% 일치
```
