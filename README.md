# LG Aimers 9기 — 끼끼팀 · 혜진 브랜치

DACON 236743 **투구 제구 성공 확률 예측** (Phase 2 온라인 해커톤).
이 브랜치는 팀 4인(`wonwoo` / `hyejin` / `sinyoung` / `chaeyeon`) 중 **혜진 작업 공간**이다.

```
최종 팀 챔피언   hj19_tmcb.zip     LB 1151.0151223813
```

---

## 1. 어디를 먼저 볼 것인가

| 문서 | 내용 |
| --- | --- |
| **[`reports/00_STATUS.md`](reports/00_STATUS.md)** | **여기부터.** 챔피언·판정 기준·닫힌 축·남은 것 — 현황 전부 |
| [`reports/11_constant_sweep_report.md`](reports/11_constant_sweep_report.md) | 라운드 1~21 (사후 상수 축 · 콜드스타트 · 트랙맨 전 경로) |
| [`reports/12_axis_reopening_report.md`](reports/12_axis_reopening_report.md) | 라운드 22~26 (축 재개방 · 판정 기준 확정 · 챔피언 재현) |
| [`REPRODUCE_HJ.md`](REPRODUCE_HJ.md) | `sy02_c4 → hj05` 재현 절차 (팀 공유용) |
| [`DIRECTION.md`](DIRECTION.md) | 2026-08-20 시점 팀 통합 방향 (역사 자료) |
| [`data_description.md`](data_description.md) | 제공 데이터 명세 |

모델 상세는 각 브랜치 문서에 있다 — `wonwoo:MODEL.md` · `sinyoung:model.md`.

---

## 2. 챔피언 계보 — 네 사람의 층이 겹쳐 있다

```
1084.1432   원우 s29        피처 파이프라인 + 코어 학습 코드
   +30.87   혜진            사후 튜닝층 (라운드 1~12)
1115.0176   p_fix_45e9
    +1.77   채연            F 전문가 + T1 패키징
1116.7884   T1
    +2.70   원우            s41 코어증강 → s43 CatBoost → s45 num_leaves×2
1119.4895   s45
    +7.67   신영            pitcher_id 임베딩 4 → 8
1127.1558   sy02_c4
    +0.97   혜진            pitcher_id 8 → 12
1128.1287   hj01_dim12
              ↓ 차원 대신 콜드스타트로 갈아탄다
    +3.32   혜진            emb0[0] ← 학습 임베딩 평균 (dim8 위에서)
1131.4503   hj03_dim8_cold
    +0.79   혜진            R/F 라우팅 t −0.25 → −0.70
1132.2410   hj05_r047f330
    +8.68   팀원            asof δ 축 gap_reverse β −0.017
1140.9183   hj09_grbc
    +4.06   원우            col 축 asof_pitcher_middle_rate_sm β −0.0070
1144.9773   hj15_pmid
    +3.24   원우            col 축 asof_pitcher_success_rate β +0.0085
1148.2205   hj16_psucc
    +2.79   원우            TrackMan CatBoost 잔차 브랜치 (γ 0.10, R∧covered)
1151.0151   hj19_tmcb                                              ← 최종
```

곁가지: `hj16_psucc_sy12mf_fsorth15` 1149.0339 (F행 직교 잔차층 +0.81) — 최종 계보에는
안 들어갔다. 롤백: `hj17_pmidsat` −11.71 · `hj13_gbmid_mf` −2.33 · `hj11_c4b175` −1.25 ·
`hj10_gstrike` −0.58 · `hj07_tmresid` −5.93.

---

## 3. 이 브랜치가 만든 것

### 점수

| 기여 | 크기 |
| --- | ---: |
| 확률공간 레벨 축 (아핀 center/scale/shift) | +18.48 |
| `β hand` · `gap_ball` · `hxc` 등 사후 상수 | ~+12 |
| **콜드스타트 `emb0[0]` 패치** | **+3.3~4.3** |
| `pitcher_id` 임베딩 8 → 12 | +0.97 |
| R/F 라우팅 `t` −0.25 → −0.70 | +0.79 |

**콜드스타트가 구조 결함 수정이다** — 매 시즌 행의 14~22%가 그 전에 없던 투수인데,
그 행 전부가 **학습 중 한 번도 갱신되지 않은 `index 0` 벡터 하나**를 공유하고 있었다
(알려진 투수 1명당 0.1% vs 미지값 1칸당 20% = **200배 과부하**).

### 판정 기준 (라운드 25 확정)

> **1층 빌드 검증은 전부 통과해야 한다. 2층은 `E[Δ]` 로 순위를 매긴다.**
> **LB 앵커는 문을 여는 열쇠가 아니라 문을 닫는 자물쇠다.**

증거 유형이 결과를 가른다 — 표본 7건, 섞인 사례 0건.

```
LB 앵커 닫힌식   3건  오차 7.4% / 상한충족 / 4.7%
홀드아웃 기반    0/4  배율 0.18 ~ -2.09,  부호 2회 오류
```

그리고 **클래스별 슬롯 효율**이 게이트를 뒤집었다.

| 클래스 | 슬롯 | 합계 | 슬롯당 |
| --- | ---: | ---: | ---: |
| C 모델 용량·구조 | 4 | +10.79 | **+2.70** |
| B 신규 δ 축 | 5 | +13.07 | **+2.61** |
| A 사후 상수 (앵커 가능) | 7 | **−0.79** | **−0.11** |
| D 학습된 층 | 1 | −5.93 | −5.93 |

**예측 가능성과 크기가 역상관이다.** 사후 상수 축은 이미 최적 근처라 앵커를 만들 수
있고 헤드룸이 없다. 새 통로는 헤드룸이 크고 그래서 앵커가 있을 수 없다.

### 반복해서 값을 낸 발견

- **in-sample 잔차는 방향을 반대로 가리킨다.** 챔피언이 학습한 시즌의 잔차는 진짜
  구조가 아니라 과적합 잔재다 — 라우팅(부호 반대·19.7배) · C1(위약 이하) ·
  컬럼 δ 축(배포된 두 축 모두 부호 반대)에서 세 번 확인.
- **닫힌 축은 층이 바뀌면 다시 열린다.** 재개방 이득 누적 **+21.67**.
- **남의 종결을 인용하기 전에 그 실험의 팔 정의를 직접 연다.** `emb8`이 4중 교란
  실험으로 닫혀 있었고, 한 칸만 바꾸니 +7.67이었다.

---

## 4. 도구 — `tools/`

### 관문

| 도구 | 하는 일 |
| --- | --- |
| **`gate_check.py`** | **슬롯 관문 — 1층 8항목 자동화 + 2층 답 강제. 제출 전 필수** |
| `verify_zip.py` | 격리 실행 · 출력 형식 · 행 독립성 · 추론 시간 |
| `fix_meta.py` | `meta.json` 감사·재작성 (**반드시 마지막에**) |
| `scan_confounded_arms.py` | 교란 팔 AST 전수 스캔 |

### 빌더 (전부 재학습 없음 · 변경 격리 검사 + SHA-256)

| 도구 | 하는 일 |
| --- | --- |
| `rebase_routing.py` | R/F 라우팅 재적용 + 레벨 재보정 |
| `rebase_col_axis.py` | 컬럼 선형 δ 축 (`art['col']`, 원우 원본에서 임시경로 1줄만 수정) |
| `add_asof_axis.py` | `asof` δ 축 추가 (기존 축 재현 게이트 G0 포함) |
| `set_delta_beta.py` | ctx/asof β 변경 + 레벨 재보정 |
| `build_coldstart_zip.py` | `index 0` 패치 |
| `build_mlp_zip.py` | MLP 멤버 교체 |
| `relevel_zip.py` · `patch_predict_scale.py` | 레벨·아핀 |

### 측정

| 도구 | 하는 일 |
| --- | --- |
| `predict_lb.py` | 순수 레벨 이동의 `ΔScore` (레벨 항등식). **아니면 예측을 거부한다** |
| `route_curvature.py` | 라우팅 축 G1~G4 (선형성 검증 · 라벨없는 `C` · LB 앵커 · 봉투) |
| `coldstart_probe.py` | 마스킹 실험 (UNK vs COLD vs TRUE) |
| `tm_features.py` · `tm_align_targets.py` | 트랙맨 집계 피처 · 투구 단위 정렬 |
| `train_mlp_dim.py` | MLP `emb4` 재학습 (`--dim`, `--unk-dropout`) |

> ⚠️ 폐기 — `col_screen.py`(in-sample 잔차 화면, G0 실패) · `rebase_c1.py`(C1 이식, 기각)

---

## 5. 저장소 구조

```
data/            대회 제공 원본 (train.csv · test.csv · trackman_history.csv · sample_submission.csv)
src/             피처·평가·엔트리포인트
tools/           관문 · 빌더 · 측정 (위 표)
reports/         라운드별 실험 기록 (00_STATUS 가 머리)
notebooks/       초기 EDA·CV (01~09 단계)
experiments/     그 단계의 스크립트판
outputs/         제출 zip (gitignore)
base/champion/   현 챔피언 풀어둔 작업본 (gitignore)
```

---

## 6. 채점 구조 (판단의 전제)

```
score = 1e5 × (1 − Brier / (r(1−r)))     r = 평가 라벨 평균 ≈ 0.486987
K = 1e5 / (r(1−r)) = 400,271
```

- **Public = 테스트 100%**, Private 은 종료 시점의 Public. **shake-up 이 없다.**
- **max 채점** — 하방은 점수가 아니라 **슬롯 한 장**이다. 그래서 목적함수는
  `P(Δ>0)` 이 아니라 `E[Δ]` 다.
- 레벨 항등식 `ΔScore = −K[2Δ·b + Δ²]` (`b = p̄ − r`) 는 **순수 레벨 이동에 한해 정확**하다.
  라운드 4에서 실측과 소수 넷째 자리까지 맞았다.

---

## 7. 환경

```
python 3.11 · numpy 2.x (필수 — 1.x 는 artifacts.joblib 을 못 읽는다)
requirements.txt 7개: numpy · pandas · scipy · scikit-learn · lightgbm · catboost · joblib
```

추론 한도 600초. 챔피언 실측 68초.
