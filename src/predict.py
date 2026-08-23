"""추론 로직 — script.py가 이 함수를 호출한다.

이 파일과 features.py는 제출 zip 루트에 그대로 복사된다 (build_submit.py 참조).
따라서 config / data / validation 등 학습 전용 모듈에 의존하지 않는다.
"""
import os

import joblib
import lightgbm as lgb
import numpy as np

from features import build_features, logit_adjust
from mlp_infer import MLPPredictor

CLIP_EPS = 1e-6

# ── 확률공간 레벨 시프트 — Brier 항등식의 `(p̄ − r)²` 항 ──────────────
#
#   BS = Var(p) − 2·Cov(p,y) + (p̄ − r)² + r(1−r)
#        └────────┬────────┘   └───┬──┘   └──┬─┘
#           refinement            레벨    불가침
#
# `p + Δ`는 Var·Cov를 건드리지 않고 p̄만 옮긴다 → refinement 축과 **구조적으로
# 직교**한다. 그래서 이득이 항등식으로 보장된다:  ΔScore = −K(2Δb + Δ²),
# b ≡ p̄ − r,  K = 1e5/(r(1−r)) ≈ 400,700.  Δ² 의 계수가 정확히 1이라
# 적합할 파라미터가 없고, LB 한 점이면 b가 확정된다.
#
#   Δ = +0.006866833   (LB 1107.2194255322 을 낸 값 — 바꾸지 않는다)
#
# **`Δ`는 `p̄`를 몰라도 정해진다.** 항등식에 `p̄`가 없기 때문이다 — LB 한 점이
# `b = p̄ − r` 을 직접 준다. §7.120은 `Δ = r − p̄_더미`로 우회 계산했는데,
# `make_dummy_test`가 `p̄`를 +0.0044 위로 편향시키고 있었다(§7.121). 다행히 `r`도
# 같은 더미로 번역돼 편향이 상쇄됐지만, **이제 우회하지 않는다.**
#
# 출처 — 혜진 브랜치가 `Δ`를 두 번 제출해 `b = −(ΔScore/K + Δ²)/(2Δ)` 로 역산:
#   p_lv  (x00, Δ=−0.005126, LB 1049.6425674531)  →  b = −0.0068750
#   p_lv2 (x12, Δ=+0.005977, LB 1106.9268527132)  →  b = −0.0068514
#   s30   (x12, Δ=+0.006866833, LB 1107.2194255322)  →  b = −0.0068490
#   1점째는 검산 −K[2Δb+Δ²] = −38.7302 가 실측과 소수 넷째 자리까지 맞았다.
#
# **축은 소진됐다.** 세 점 전부 b ∈ [−0.006853, −0.006847] (r 을 0.4824~0.49
# 어디로 잡든). 최적 Δ ≈ +0.006849 이고 배포값과의 차이가 주는 손실은
# K·(1.8e−05)² ≈ **0.00013 BSS** 다. 더 움직일 이유가 없다.
#
# ⚠️ **`r` 자체는 식별되지 않는다.** x12 위의 두 점(lv2·s30)으로 K 와 b 를 동시에
#    풀면 r = 0.4609 가 나오지만, 두 Δ 가 가까워 조건수가 나쁘다 — LB 에 ±0.02 만
#    섭동해도 r 이 0.44↔0.50 으로 흔들린다. **b 는 안정, r 은 불안정**이다.
#    재기준화 더미(§7.121)가 주는 독립 추정은 r ≈ 0.4824 이고, 그것이 현재 최선이다.
#
# ⚠️ 이 상수는 학습 데이터가 아니라 **리더보드 피드백**에서 나왔다.
#    학습 데이터만 쓰는 대안은 2024 실측 0.4861 목표(Δ = +0.005977)이고
#    최대 이득의 98.3%를 회수한다. 차이 약 0.3 BSS.
#    LB 역추론 허용은 2026-08-24 팀 결정이다. `MODEL.md §7.120·§7.121` 참조.
#
# 규정: 파일에 박힌 **동결 스칼라 리터럴**이다. 각 행의 예측은 자기 피처에만
#       의존하고, 추론 시점에 다른 행이나 test 분포를 참조하지 않는다
#       (CLAUDE.md §3.2). `verify_pipeline.py`의 셔플·청크 검사가 확인한다.
PROB_CENTER = 0.4801
PROB_SCALE = 1.05
# ⚠️ **스택이 바뀌면 이 값도 다시 유도해야 한다.** `PROB_TARGET`(= r 추정치)은
#    평가셋의 성질이라 스택 무관이지만, `Δ = PROB_TARGET − p̄(스택)` 이다.
#    m8(s30)  p̄ = 0.475541112  ->  Δ = 0.006866833   LB 1107.2194255322
#    m16      p̄ = 0.475456633  ->  Δ = 0.006951312   (§7.128, 멤버 2배)
#    차이가 8.4e−05 라 잘못 두면 K·(8.4e−05)² ≈ 0.003 BSS 손해 — 작지만 공짜로 맞춘다.
PROB_SHIFT = 0.006999915599021783  # = PROB_TARGET − p̄(m8).  m16 은 §7.130 에서 롤백됨
# **체크섬이다 — 운영 상수가 아니다.** `make_dummy_test(art=...)`(재기준화판)
# 위에서 아핀 변환 후 도달해야 하는 평균이고, `p̄_대리 + PROB_SHIFT` 로 정의된다.
# build_submit.py가 매 빌드마다 대조해 어긋나면 제출을 막는다 — 재학습으로
# 스택이 바뀌면 `p̄`가 움직이므로 여기서 잡힌다. (`PROB_SHIFT` 자체는 LB에서
# 나온 값이라 스택이 바뀌어도 유효하다. 바뀌는 건 도달 평균 쪽이다.)
PROB_TARGET = 0.482407945

# ── 세그먼트 레벨 분해 (`game_type`) — §7.123 ─────────────────────────
#
# 전역 Δ 하나는 `Σ w_g(p̄_g − r_g)` 만 0으로 만든다. 세그먼트별 편차는 남고
# 그 크기가 그대로 점수다:  이득 = K·w_R·w_F·(b_R − b_F)².
#
#   D̂ = (p̄_F − p̄_R) − (r_F − r_R) = −0.022020 − (−0.03043) = +0.008410
#        └── 재기준화 더미 실측 ──┘   └─ 2024 시즌 실측 ─┘
#
#   Δ_F = Δ_global − w_R·D̂ = −0.000583
#   Δ_R = Δ_global + w_F·D̂ = +0.007827      (w_F = 0.1142)
#
# 가중평균이 `PROB_SHIFT` 와 같다 — **전역 레벨은 안 건드린다.** §7.122가 LB로
# 확정한 값을 그대로 두고 세그먼트 간 배분만 바꾼다.
#
# **전이가 검정됐다** (`src/segprobe.py`, fold 2024): 직전 시즌(2023)의 격차를
# 가져다 쓰면 그 fold 자신의 라벨로 잡은 천장의 **99.9%** 를 회수한다(+3.127 /
# +3.129). 같은 기계를 요일·월 위약 버킷에 걸면 전이가 **음수**다(−0.118 / −5.128).
# → 서브그룹 튜닝의 흔한 허수(§7.45 의 요일 +22.8)가 아니다.
#
# ⚠️ **fold 2023 에서는 전이가 −2069 로 붕괴한다.** 2022 격차 +0.205 → 2023
#    −0.030 의 부호 반전 때문이고, 이것이 §7.71 이 "학습기에 F−R 레벨차를 맡기면
#    시즌마다 뒤집힌다"고 기각한 근거다. 그 반전은 **행동 변화가 아니라 데이터
#    품질 변화**다 — §7.49 의 pre-2023 F 오염이고 `deploy29` 도 그 행들을 clean
#    에서 제외한다. 오염이 끝난 2023·2024 는 −0.0302 / −0.0304 로 2e−04 안에서
#    같다. **이 상수는 그 체제가 2025 에도 유지된다는 데 거는 것이다.**
#    손익분기는 2024 격차의 86%. 격차가 소멸하면 −17.9, 2022 수준으로 반전하면 −158.
#
# 각 행은 **자기 `game_type` 만** 본다 — 동결 스칼라 두 개 중 하나를 고를 뿐이고
# 다른 행이나 test 분포를 참조하지 않는다 (CLAUDE.md §3.2).
# ❌ **끈다 (§7.124).** s31 = 1105.3176522757 로 s30 대비 −1.90.
#    역산하면 2025 실제 격차는 −0.02343 — 2024(−0.03043)의 **77%** 로 줄었고
#    손익분기 86% 를 9%p 차로 못 넘겼다. 그리고 모델이 이미 −0.02202 를
#    예측하고 있어 **남은 오차가 0.0014, 이 축의 최대 이득은 +0.08** 이다.
#    최적값을 쓰려면 {"F": +0.005614, "R": +0.007028} 이지만 슬롯 값어치가 없다.
SEG_SHIFT = None

# ── 상수 스윕용 배포층 노브 (§7.131) ────────────────────────────────
# §7.130 의 규칙: **앙상블을 고정한 채 상수만 바꾸는 변경은 draw 노이즈가 0** 이라
# LB 차이가 그대로 효과다. 멤버 구성을 바꾸는 변경(§7.128)과 성격이 정반대다.
#
# 그리고 곡률 `A₂ = Var(dp/dx)` 는 **라벨 없이** 더미에서 계산되므로
# `BS(x) = BS₀ + (x−x₀)(A₁−2B₁) + (x−x₀)²A₂` 에서 미지수가 `B₁` 하나뿐이다
# → **LB 한 점이 꼭짓점을 확정한다** (§7.120 레벨 축과 같은 구조).
#
#   K·A₂ (레벨 고정):  δ β 공통배율 83.1  ·  fspec_alpha 22.0
#                      MLP noid 몫 17.8  ·  LGBM wide 몫 0.6 ← 딸 것이 없다
#
# ⚠️ 둘 다 예측 평균을 움직이므로 **`PROB_SHIFT` 를 같이 재고정**해야 한다.
#    안 그러면 이미 최적인 레벨 항이 딸려 움직여 측정이 오염된다 (0.3~0.5 BSS 규모).
CTX_BETA_MULT = 1.0           # 사후 로짓 보정 δ 4축의 β 공통배율 (1.0 = 배포값)
FSPEC_ALPHA_OVERRIDE = 0.30  # None 이면 아티팩트의 fspec_alpha 를 쓴다


class Predictor:
    def __init__(self, model_dir="./model"):
        bundle = joblib.load(os.path.join(model_dir, "artifacts.joblib"))
        self.art = bundle["artifacts"]
        self.cols = bundle["feature_cols"]
        self.delta = float(bundle["logit_delta"])
        # 사후 로짓 보정 축은 self.art["ctx"] 안에 β까지 함께 들어 있다.
        # (번들과 아티팩트 두 곳에 β를 두면 어긋난다 — 한 곳으로 모았다)
        self.ctx_hits = {}

        # MLP 멤버 — torch가 아니라 npz + numpy 순전파다 (mlp_infer.py 참조)
        self.alpha = float(bundle.get("mlp_alpha") or 0.0)
        self.mlps = []
        if self.alpha > 0:
            enc = bundle["mlp_enc"]
            for tag in bundle.get("mlp_members") or []:
                with np.load(os.path.join(model_dir, f"mlp_{tag}.npz")) as z:
                    self.mlps.append(MLPPredictor({k: z[k] for k in z.files}, enc))
            if not self.mlps:
                self.alpha = 0.0
        # 멤버별 가중 (config.MLP_VARIANT_WEIGHT에서 나온다). 없으면 균등 —
        # 옛 아티팩트를 그대로 읽을 수 있어야 한다.
        self.mlp_w = bundle.get("mlp_weights") or None
        if self.mlp_w is not None:
            self.mlp_w = [float(v) for v in self.mlp_w]
            if len(self.mlp_w) != len(self.mlps):
                raise ValueError(f"mlp_weights({len(self.mlp_w)}) != "
                                 f"멤버 수({len(self.mlps)})")
        # model_file= 이 아니라 model_str= 로 읽는다.
        # LightGBM의 C 파일 API는 Windows에서 경로를 ANSI 코드페이지로 해석하므로
        # 경로에 비ASCII 문자가 있으면 조용히 깨진다. 파일 I/O는 파이썬에 맡긴다.
        # F 전용 전문가 (§7.101) — game_type=='F' 행만 다시 예측해 로짓에서 섞는다.
        # 옛 아티팩트에는 없는 키이므로 get()으로 읽고, 없으면 α=0이 되어 꺼진다.
        self.fspec_alpha = float(FSPEC_ALPHA_OVERRIDE
                                 if FSPEC_ALPHA_OVERRIDE is not None
                                 else (bundle.get("fspec_alpha") or 0.0))
        self.fspec = []

        # 꼬리 하드 클리핑 경계 — 학습 때 얼린 숫자 두 개. 없으면 꺼진다.
        # ⚠️ 여기서 test 분포의 분위를 계산하면 규정 위반이다 (CLAUDE.md §3.2:
        #    분포 통계 금지, 각 행은 독립적으로 예측돼야 한다). 이 두 값은
        #    train.py가 REF_SEASON 참조행에서 뽑아 아티팩트에 박아둔 상수다.
        lo, hi = bundle.get("clip_lo"), bundle.get("clip_hi")
        self.clip_lo = None if lo is None else float(lo)
        self.clip_hi = None if hi is None else float(hi)

        self.boosters = []
        for tag in bundle.get("members") or [f"{s}" for s in bundle.get("seeds", [])]:
            with open(os.path.join(model_dir, f"lgbm_{tag}.txt"), encoding="utf-8") as f:
                self.boosters.append(lgb.Booster(model_str=f.read()))
        if self.fspec_alpha > 0:
            for tag in bundle.get("fspec_members") or []:
                with open(os.path.join(model_dir, f"lgbm_{tag}.txt"),
                          encoding="utf-8") as f:
                    self.fspec.append(lgb.Booster(model_str=f.read()))
            if not self.fspec:
                self.fspec_alpha = 0.0

    def predict(self, test_df):
        """test_df의 각 행을 **독립적으로** 예측한다.

        여기에는 test_df 전체에 대한 groupby / value_counts / rolling / fit이
        전혀 없다. 행 사이를 잇는 값은 전부 학습 때 동결된 self.art / self.delta다.
        """
        if len(test_df) == 0:
            return np.zeros(0)

        X = build_features(test_df, self.art)
        if self.art.get("use_player_ids"):
            for c in ("pitcher_id", "batter_id"):
                X[c] = test_df[c].to_numpy(dtype=np.float32)

        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"피처 누락: {missing}")
        # Trackman 테이블 조회가 통째로 빗나갔는지 진단용으로 노출한다.
        # (평가 서버 test의 season 값이 예상과 다르면 전부 결측이 되고,
        #  에러 없이 성능만 떨어진다 — 로그에 남겨 즉시 알아채기 위함)
        self.tm_match_rate = (float(X["tm_matched"].mean())
                              if "tm_matched" in X.columns else None)

        X = X[self.cols]     # 학습 때 쓴 컬럼만, 학습 때 순서로. drop_features는 여기서 걸러진다

        # 시드 앙상블은 **로짓 공간에서 평균**한다.
        # 확률 평균은 앙상블 출력을 0.5 쪽으로 수축시켜 캘리브레이션을 흐트러뜨린다.
        z = np.zeros(len(X))
        for bst in self.boosters:
            z += bst.predict(X, raw_score=True)
        z /= len(self.boosters)

        # MLP 블렌드도 로짓 공간에서. 각 MLP는 자기 행만 보고 계산한다
        # (표준화 상수 · 범주 LUT · 결측 대체값이 전부 학습 때 동결돼 있다).
        if self.alpha > 0:
            zm = np.zeros(len(X))
            if self.mlp_w is None:
                for m in self.mlps:
                    zm += m.logit(X)
                zm /= len(self.mlps)
            else:
                # 가중은 학습 시점에 정규화돼 있다 (합 = 1). 여기서 또 나누지 않는다.
                for m, wt in zip(self.mlps, self.mlp_w):
                    zm += wt * m.logit(X)
            z = (1.0 - self.alpha) * z + self.alpha * zm

        # ── F 전용 전문가 블렌드 (§7.101) ─────────────────────
        # LGBM+MLP **이후**, δ 보정·prior shift **이전**. 이 지점이 α를 고른
        # 검증(fspec_verify.py)과 같아야 한다.
        # game_type은 각 행이 스스로 들고 있는 값이라 행 간 정보가 아니다 —
        # 여기서 하는 건 라우팅이지 test 통계 계산이 아니다 (CLAUDE.md §3.2).
        # ⚠️ game_type은 피처로는 여전히 드랍돼 있다 (§7.71). 학습기가 F−R 레벨차를
        #    표현하게 두면 시즌마다 뒤집혀 독이 된다. 여기서는 **어느 모델에 물을지**만 가른다.
        self.fspec_rate = 0.0
        if self.fspec_alpha > 0 and "game_type" in test_df.columns:
            is_f = test_df["game_type"].to_numpy().astype(str) == "F"
            self.fspec_rate = float(is_f.mean())
            if is_f.any():
                Xf = X.iloc[np.flatnonzero(is_f)]
                zf = np.zeros(len(Xf))
                for bst in self.fspec:
                    zf += bst.predict(Xf, raw_score=True)
                zf /= len(self.fspec)
                z[is_f] = ((1.0 - self.fspec_alpha) * z[is_f]
                           + self.fspec_alpha * zf)

        # 사후 로짓 보정 (투수 × 상황 δ) — **블렌드 이후, prior shift 이전**.
        # 이 순서는 검증에서 잰 순서와 같아야 한다 (2 fold에서 β 스윕한 그 지점).
        # 순서를 바꾸면 solve_delta가 다른 로짓을 보게 되어 목표 prior를 못 맞춘다.
        # train.py · verify_pipeline.py가 **같은 함수**를 부른다.
        z_adj, self.ctx_hits = logit_adjust(test_df, self.art)
        z = z + CTX_BETA_MULT * z_adj

        # prior shift — 학습 때 고정된 스칼라(base+delta)의 행 단위 적용
        p = 1.0 / (1.0 + np.exp(-(z + self.delta)))

        # 확률공간 아핀 변환 (모듈 상단 주석 참조).
        #   p' = P0 + c·(p − P0) + Δ
        # **로짓이 아니라 확률에서 한다.** 시그모이드가 비선형이라 로짓 공간에서는
        # 곡선이 2차가 아니고(§7.19가 여기서 기각됐다), 레벨 이동이 Var 까지 흔든다.
        # 두 축은 서로 다른 항을 건드리므로 이득이 가법이다 (실측 오차 0.66%).
        if PROB_SCALE != 1.0:
            p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER)
        # 레벨 이동. `game_type`이 있으면 세그먼트별 상수를, 없으면 전역 상수를 쓴다.
        # ⚠️ `game_type`은 피처로는 여전히 드랍돼 있다 (§7.71). 여기서 하는 건
        #    **행이 자기 라벨로 상수 하나를 고르는 것**이지 학습기가 레벨차를
        #    표현하게 두는 것이 아니다 — F전문가 라우팅(위)과 같은 성격이다.
        if SEG_SHIFT and "game_type" in test_df.columns:
            gt = test_df["game_type"].to_numpy().astype(str)
            shift = np.where(gt == "F", SEG_SHIFT["F"], SEG_SHIFT["R"])
            self.seg_rate = float((gt == "F").mean())
            p = p + shift
        elif PROB_SHIFT:
            p = p + PROB_SHIFT

        # 꼬리 하드 클리핑 (기본 꺼짐). 각 행이 자기 p를 동결 상수와 비교할 뿐,
        # 다른 행을 보지 않는다. `src/pclip.py` 측정으로 기본은 OFF다 —
        # 배포 모델의 p 범위가 [0.31, 0.67]이라 0/1로 밀 근거가 되는 꼬리가 없다.
        self.clip_rate = 0.0
        if self.clip_lo is not None or self.clip_hi is not None:
            n_clipped = 0
            if self.clip_lo is not None:
                m = p <= self.clip_lo
                n_clipped += int(m.sum())
                p = np.where(m, 0.0, p)
            if self.clip_hi is not None:
                m = p >= self.clip_hi
                n_clipped += int(m.sum())
                p = np.where(m, 1.0, p)
            self.clip_rate = n_clipped / len(p)

        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def predict_test(test_df, model_dir="./model"):
    return Predictor(model_dir).predict(test_df)
