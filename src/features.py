"""피처 빌더 — 학습과 추론의 **단일 진실 소스**.

리키지 규칙 (CLAUDE.md §3.2):
    이 모듈의 모든 함수는 **행 단위로 독립적**이다.
    `df` 전체에 대한 groupby / value_counts / rolling / 분위수 계산이 없다.
    행 사이를 잇는 값은 전부 `artifacts` 딕셔너리에서 온다 —
    이 딕셔너리는 **train에서만 fit** 되고 `model/`에 직렬화되어 추론 시 조회만 된다.

    → 그래서 `fit_artifacts(train_df)`와 `build_features(df, artifacts)`가 분리돼 있다.
      `build_features`는 절대 fit 하지 않는다.
"""
import numpy as np
import pandas as pd

import contam            # 오염 보정 조회 (학습/추론 공용). 제출 zip에 같이 들어간다.

# 이 모듈은 config를 import하지 않는다 — 제출 zip에 그대로 복사되어
# 추론 시에도 동일한 코드가 돌아야 하기 때문. 튜닝 상수는 artifacts에 동결된다.

# ── 스무딩 대상: (rate 컬럼, 표본수 컬럼) ──────────────────────────
SMOOTH_SPEC = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]

_PITCHER_N_COLS = {"asof_pitcher_n", "asof_pitcher_pitchmix_n"}

# ── (시즌 × game_type) 중심화 — MODEL.md §7.106 ────────────────────────
# `season`·`game_type`은 모델 피처가 아니다(DROP_FEATURES). 그런데 `asof_*`의 리그
# 평균은 시즌마다 컬럼 std의 0.75~1.12배씩 움직인다. 그래서 같은 asof=0.52가
# 2019엔 평균 이하, 2024엔 평균 이상인데 모델은 둘을 구분할 수단이 없다.
#
#   측정된 결과는 '감쇠'가 아니라 **증폭**이었다 (asofctr.py B):
#     풀링 기울기 / 시즌내 기울기 = 0.43~1.30. 타자 컬럼은 2배 이상 부풀려져 있다.
#     피처와 타깃이 같은 방향으로 드리프트하니 시즌 간 공분산이 기울기에 덧붙는다.
#   → **test에는 시즌이 하나뿐이라 그 성분은 존재할 수 없다.** 편향으로만 남는다.
#
# 변환:  v ← v − μ(season, game_type) + μ_global
#   전역 평균을 다시 더하는 이유는 스무딩 prior·`form_*`·`matchup_*`의 스케일을
#   그대로 두기 위함이다. 트리에는 상수항이 무의미하므로 "평균 0"과 동치다.
#
# ⚠️ 2025(및 각 fold의 검증 시즌)는 표에 없다 → **외삽**한다. fold 검증 시즌이
#    항상 미지라서, fold 실험이 외삽 오차까지 같이 평가한다. 설계상 그렇게 뒀다.
# (대상 컬럼 목록 CENTER_COLS 는 PREV_COLS 정의 뒤에 있다.)

# 범주형 → 정수 코드 (미지값 -1).
# str/int을 분리해 두는 이유: train은 dtype을 명시해 읽지만 평가 서버의 test는
# pandas 기본 추론으로 읽힐 수 있다. int16 vs int64가 섞이면 dict 조회가 조용히
# 전부 실패해 -1로 떨어진다 — 에러 없이 성능만 죽는 종류의 버그다.
CATEGORICAL_STR = ["base_state", "top_bottom", "game_type"]
CATEGORICAL_INT = ["pitcher_team_id", "batter_team_id"]

# ── [실험 전용] asof 곱 상호작용 (MODEL.md §7.120) ────────────────────
# 기본값 None → 아무 컬럼도 안 만든다. **배포 경로는 비트 단위로 그대로다.**
# 실험 스크립트가 `features.EXTRA_PROD = "prod"` 처럼 세워서 켠다.
#   "prod"     asof_pitcher_reverse_rate x asof_pitcher_success_rate
#              (사용자 제안. 두 주효과는 이미 배포 피처라 base 가 곧 주효과 대조군)
#   "placebo"  같은 success_rate x **죽은 축**(요일)  — 영가설.
#              요일은 §7.78 에서 죽은 것이 확인된 실축이고, 공유항(success_rate)을
#              남기므로 "컬럼 하나 더 붙은 효과"만 분리해 준다 (memory: 위약은
#              공유 항을 남겨야 한다 / 죽은 축을 영가설로 쓰라).
# 두 팔 모두 **같은 이름·같은 개수·같은 NaN 패턴**의 컬럼을 만든다.
EXTRA_PROD = None

# 요일(0~6)을 reverse_rate 의 5~95 퍼센타일 구간으로 늘린 값. 스케일을 맞춰야
# 위약이 후보와 같은 프로파일을 갖는다.
_DOW_LO, _DOW_HI = 0.119048, 0.307958

# ── [실험 전용] 이력 깊이 / prev 결측 채움 (MODEL.md §7.133) ──────────
# 기본값 None → 배포 경로는 비트 단위로 그대로다.
#
# 왜 있나. `prev{1,3,5}_game_*` 는 **완료 경기 수가 적으면 서로 같은 값이 된다** —
# 경기 1개면 "직전 1/3/5경기"가 전부 그 한 경기다. 그래서 `form_p3_minus_p5` 가
# 구조적으로 정확히 0 인 행이 train 의 **5.89%**(86,820행) 있다. 모델은 그것을
# "폼 변화가 없다"와 구별할 수단이 없다 — `miss_prev_game` 플래그는 완료 경기
# **0개**(1.98%)만 덮는다. 결측 표현이 이진인데 실제 조건은 서수다.
#
#   "depth"     `hist_depth` 서수 1개 추가 (0=이력없음 1=1경기 2=2~3 3=4+).
#               그 행 **자신의 컬럼끼리만 비교**해서 나온다 — groupby 없음,
#               아티팩트 없음 (CLAUDE.md §3.2 행 독립성이 자명하다).
#   "depthnan"  위 + 구조적 0 을 NaN 으로 바꾼다. LGBM 은 결측 분기로 자동 분리,
#               MLP 는 `nan_plan` 밖 컬럼이라 0 으로 떨어져 사실상 현행 유지 →
#               **두 학습기가 다르게 반응한다** (§7.92 형 반전 위험을 일부러 잰다).
#   "carry"     prev 결측을 리그 prior 가 아니라 **그 투수 자신의 커리어값**으로
#               채운다 → form_* 이 정확히 0 이 된다. 현행은 `prior - career_i` 라
#               커리어에 대한 **가짜 단조감소 기울기**가 1.98% 행에 깔린다.
#   "depthflag" §7.133 후속. **컬럼을 늘리지 않고** 기존 `miss_prev_game` 을 서수로
#               승격한다 — `depth` 팔은 내용(위약 대비 +3.6/+0.6)이 있었는데 컬럼
#               하나 추가하는 비용(-5.1/-2.6)에 먹혔다. 방향은 원래 플래그와 맞춘다
#               (1 = 이력 없음). 트리는 0.5 에서 쪼개면 원래 플래그를 정확히
#               복원하므로 **엄밀한 상위집합**이다.
HIST_MODE = None
_HIST_MODES = ("depth", "depthnan", "carry", "depthflag")
CATEGORICAL_COLS = CATEGORICAL_STR + CATEGORICAL_INT

# prev{1,3,5}_game_* 는 항상 같이 결측된다 (등판 이력 없음) → 플래그 하나로 충분
PREV_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]

# 중심화 대상 = 스무딩 대상 rate 전부 + 직전등판 rate 전부.
# prev{1,3,5}가 드리프트가 가장 크다 (−0.058, 컬럼 std의 0.45~0.79배).
CENTER_COLS = [c for c, _ in SMOOTH_SPEC] + PREV_COLS

# §7.119 — **부분 중심화** 대상 (사용자 지정 5개). 투수 자신의 제구 축만 건드리고
# 타자·구종믹스·middle/ball/strike 계열은 원본을 유지한다.
CENTER_SUBSET5 = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]

# mode 별 성질. 셋을 따로 두는 이유 — `mu5`(대조군)는 μ 컬럼만 추가하고 값은
# **안 건드린다.** 그래서 priors 를 0 으로 만들면 안 된다 (값이 여전히 .5 대역).
_SUBSET5_MODES = ("lag5", "lag5mu", "mu5", "mu5dead", "lag5sq",
                  "pow5_2", "pow5_3")                   # 5개만 대상
# §7.151 — 중심화 **없이** 원시 rate 를 p 제곱한다. p 는 하나의 노브다:
#   p=1 이 현 배포 → 용량-반응으로 읽는다 (단일 지점이 노이즈 대역이면 해석 불가).
_POW5 = {"pow5_2": 2, "pow5_3": 3}
_MU_FEATURE_MODES = ("lag5mu", "mu5", "mu5dead")        # μ 형태 컬럼 5개를 추가
_ZERO_PRIOR_MODES = ("zero", "zero_scale", "zero_off", "lag", "lag5", "lag5mu",
                     "lag5sq")

# §7.119 위약 축 — **요일**. `DROP_FEATURES` 에 있고 §7.78 에서 죽은 것이 확인됐다.
# `mu5dead` 는 μ 컬럼과 **같은 이름·같은 개수·같은 구조**(그룹 평균 조회)를 만들되
# 축만 죽은 것으로 바꾼다. 목적은 "내용이 0인 컬럼 5개를 붙이는 것 자체의 비용"을
# 재는 것 — §7.120 에서 컬럼 1개 위약이 후보와 **같은 크기로 움직였다**(둘 다 구간이
# 0 을 포함해 부호 미확정). 그래서 그 양은 미지수이고 여기서 직접 잰다.
# (MEMORY: 위약은 공유 항을 남겨야 한다 / 죽은 축을 영가설로 쓰라 / 노이즈 바닥은
#  학습기마다 다르다 → 1개에서 잰 값을 5개로 외삽하지 않고 직접 잰다.)
# ⚠️ 한계: 요일은 train·test 에 같은 수준이 전부 나타나므로 이 위약은 μ 의
#    **범위 밖 외삽** 성분은 재현하지 않는다. 컬럼 수 성분만 격리한다.
_DEAD_COL = "game_dayofweek"


def fit_asof_center(train_df, cols=CENTER_COLS, min_n=500):
    """(시즌 × game_type) → 평균·표준편차 표 + **다음 시즌 외삽** 규칙. train에서만 호출."""
    ss = train_df["season"].to_numpy(dtype=np.int64)
    gg = train_df["game_type"].astype("object").to_numpy().astype(str)
    ug = np.unique(gg)
    out = {"table": {}, "fallback": {}, "global": {}, "sd": {}, "sd_fb": {}, "sd_global": {},
           "lag": {}, "lag_fb": {}}
    for col in cols:
        v = train_df[col].to_numpy(dtype=np.float64)
        fin = np.isfinite(v)
        out["global"][col] = float(v[fin].mean()) if fin.any() else 0.5
        out["sd_global"][col] = float(v[fin].std()) if fin.any() else 1.0
        tab, fb, sdt, sdf = {}, {}, {}, {}
        lag, lfb = {}, {}
        for gtv in ug:
            pts, spts, cnt = [], [], []
            for yy in sorted(set(ss[gg == gtv].tolist())):
                m = (ss == yy) & (gg == gtv) & fin
                if m.sum() >= min_n:
                    tab[(int(yy), gtv)] = float(v[m].mean())
                    sdt[(int(yy), gtv)] = max(float(v[m].std()), 1e-6)
                    pts.append((int(yy), float(v[m].mean())))
                    spts.append((int(yy), max(float(v[m].std()), 1e-6)))
                    cnt.append(int(m.sum()))
            for src, dst in ((pts, fb), (spts, sdf)):
                if len(src) >= 2:
                    # 연속 차분의 평균으로 1스텝 외삽 (TARGET_PRIOR_2025와 같은 규칙)
                    d = float(np.mean(np.diff([p[1] for p in src])))
                    dst[gtv] = (src[-1][0], src[-1][1], d)
                elif src:
                    dst[gtv] = (src[-1][0], src[-1][1], 0.0)
            # mode="lag" (§7.117) — 시즌 s 의 기준 = s **이전** 시즌들의 행수 가중
            # 누적 평균. 첫 시즌은 과거가 없어 자기 시즌 평균. 표에 없는 시즌
            # (fold 검증 시즌·2025)은 전체 누적을 slope 0 fallback 으로 — **외삽이
            # 없다.** 학습·추론 모두 기준이 한 발 늦어 정렬 잔차의 생성 규칙이 같다
            # (기각된 네 구현의 공통점인 학습/추론 정렬 비대칭을 제거하는 설계).
            csum, cn = 0.0, 0
            for (yy, mu), n in zip(pts, cnt):
                lag[(yy, gtv)] = (csum / cn) if cn else mu
                csum += mu * n
                cn += n
            if cn:
                lfb[gtv] = (pts[-1][0], csum / cn, 0.0)
        out["table"][col], out["fallback"][col] = tab, fb
        out["sd"][col], out["sd_fb"][col] = sdt, sdf
        out["lag"][col], out["lag_fb"][col] = lag, lfb
    return out


def _center_base(art, ac, col):
    """중심화 후 값이 앉을 기준점. **스무딩 prior 와 같아야 한다.**

    ⚠️ 처음엔 `ac["global"]`(비가중 전체 평균)을 더했다가 틀렸다 (§7.106 ⑧).
       근거는 "트리에는 상수항이 무의미하니 평균 0 과 동치" 였는데, 이 컬럼은 트리로
       바로 안 가고 **먼저 `_smooth` 를 통과한다.** 스무딩은 평행이동에 불변이 아니다:

           _smooth(v, n, prior, k) = (n·v + k·prior) / (n + k)

       `prior` 는 고정인데 `v` 만 밀면 그 간격만큼 **저표본 행만 끌려간다.**
       실측: prior(recency 가중) .52686 vs global(비가중) .54018 → 간격 +0.01331.
       같은 시즌 안에서 n 에 따라 인위적 0.013 격차가 생겼고, 이는 §7.106 ⑦이 찾은
       진짜 교란(신인–베테랑 +0.035)과 **같은 방향**이라 손해를 키웠다.
    """
    p = (art or {}).get("priors", {})
    return float(p[col]) if col in p else float(ac["global"][col])


def _lookup_group(g, tab, fb, gm):
    """행별 그룹값. 표에 없는 시즌은 외삽. **행 단위 조회일 뿐이다.**"""
    ss = g["season"].to_numpy(dtype=np.int64)
    gg = g["game_type"].astype("object").to_numpy().astype(str)
    ug = np.unique(gg)
    k = ss * len(ug) + np.searchsorted(ug, gg)
    out = np.full(len(k), gm, dtype=np.float64)
    for kk in np.unique(k):                       # 조합이 십수 개뿐이다
        yy, c = int(kk // len(ug)), int(kk % len(ug))
        gtv = str(ug[c])
        val = tab.get((yy, gtv))
        if val is None:
            f = fb.get(gtv)
            val = (f[1] + f[2] * (yy - f[0])) if f else gm
        out[k == kk] = val
    return out


def _center_offset(g, ac, col):
    return _lookup_group(g, ac["table"][col], ac["fallback"][col], ac["global"][col])


def fit_dead_group(train_df, cols, min_n=500):
    """§7.119 위약 표 — (요일 × game_type) 그룹 평균. train에서만 호출.

    μ 표와 **구조가 같다**(그룹 평균 조회). 축만 죽은 것이라, 여기서 나오는 차이는
    '컬럼 5개가 붙었다'는 사실 자체의 값이다.
    """
    kk = train_df[_DEAD_COL].to_numpy(dtype=np.int64)
    gg = train_df["game_type"].astype("object").to_numpy().astype(str)
    out = {"table": {}, "global": {}}
    for col in cols:
        v = train_df[col].to_numpy(dtype=np.float64)
        fin = np.isfinite(v)
        out["global"][col] = float(v[fin].mean()) if fin.any() else 0.5
        tab = {}
        for gtv in np.unique(gg):
            for dd in np.unique(kk):
                m = (kk == dd) & (gg == gtv) & fin
                if m.sum() >= min_n:
                    tab[(int(dd), gtv)] = float(v[m].mean())
        out["table"][col] = tab
    return out


def _lookup_dead(g, tab, gm):
    """행별 조회. `_lookup_group`과 같은 형태지만 키가 (요일, game_type)이다."""
    kk = g[_DEAD_COL].to_numpy(dtype=np.int64)
    gg = g["game_type"].astype("object").to_numpy().astype(str)
    ug = np.unique(gg)
    k = kk * len(ug) + np.searchsorted(ug, gg)
    out = np.full(len(k), gm, dtype=np.float64)
    for key in np.unique(k):
        dd, c = int(key // len(ug)), int(key % len(ug))
        out[k == key] = tab.get((dd, str(ug[c])), gm)
    return out


def _center_apply(g, art, ac, col, raw):
    """중심화 적용. mode 가 설계 전체를 정한다.

    base       v − μ + priors[col]   레벨 자리를 유지 (스무딩 prior 와 맞춘다)
    zero       v − μ                 **평균 0.** 이때 `priors[col]` 은 0 으로 바뀌어 있어
                                     스무딩·결측대체·form_* 가 전부 따라온다 (§7.116)
    zero_scale (v − μ) / σ           평균 0 + 시즌별 스케일 정규화
    lag        v − μ_cum(< s)        기준이 시즌 s−1 까지의 누적 평균 (§7.117).
                                     미지 시즌은 전체 누적 상수 — 외삽이 없어 학습/추론
                                     정렬 비대칭이 사라진다. priors 는 zero 처럼 0.
    lag5       lag 를 5개 컬럼에만    §7.119. 대상 축소가 유일한 차이다.
    lag5mu     lag5 + μ 를 컬럼으로   §7.119 사용자 제안. 변환이 **가역**이 된다
                                     (모델이 v = (v−μ) + μ 를 복원할 수 있다).
    mu5        **중심화 안 함**       §7.119 짝지은 대조군 — μ 컬럼만 추가한다.
                                     "μ 가 한 일"과 "중심화가 한 일"을 가른다.
    """
    mode = ac.get("mode", "base")
    if mode in ("mu5", "mu5dead"):
        return raw                            # 값은 그대로 — μ 컬럼만 따로 나간다
    if mode in _POW5:
        # §7.151 — **중심화 없이** 원시 rate 를 p 제곱한다 (사용자 제안).
        # rate 는 [0,1] 이라 p 제곱이 강단조 = 정보 손실 없음(부호 접힘도 없다.
        # x³ 은 실수 전체에서 강단조라 중심화값에 걸어도 안전한 형태이기도 하다).
        # 그래도 무연산이 아니다 — §7.150 이 확인했듯 변환은 `_smooth` 앞에 걸려
        # `_sm`·`form_*`·`matchup_*` 로 번지고 거기서 단조성이 깨진다.
        # 실측 corr(v, v²): success .985 / reverse .919 (꼬리가 [0,1] 전체라 휜다).
        # ⚠️ priors 도 같은 지수로 올려야 한다 — 아래 fit_artifacts 참조.
        return np.power(raw, _POW5[mode]).astype(np.float32)
    if mode in ("lag", "lag5", "lag5mu", "lag5sq"):
        v = raw - _lookup_group(g, ac["lag"][col], ac["lag_fb"][col], ac["global"][col])
        if mode == "lag5sq":
            # §7.150 — **부호 있는 제곱** `sign(c)·c²` (사용자 수정안).
            # 단순 제곱은 부호를 접어 정보를 버리는데, 부호를 도로 붙이면 **강단조**라
            # 정보 손실이 없다. 그 대신:
            #   · 트리에는 **무연산**이다 (분할이 `x ≤ t` 라 순서만 보면 된다).
            #     → LGBM 예측이 lag5 와 같아야 정상. 하네스가 이걸 assert 한다.
            #   · MLP 는 단조 변환에 불변이 아니라 **꼬리가 늘어난 기하**를 본다.
            # ⚠️ 스케일 상수는 여전히 무의미하다 — sign(kc)(kc)² = k³·sign(c)c² 이고
            #    표준화가 k³ 을 그대로 나눈다 ("×100" 이 무연산인 것과 같은 이유).
            v = np.sign(v) * v * v
        return v.astype(np.float32)
    v = raw - _center_offset(g, ac, col)
    if mode in ("zero", "zero_off"):
        pass                                  # 평균 0 — 더하는 상수가 없다
    elif mode == "zero_scale":
        sd = _lookup_group(g, ac["sd"][col], ac["sd_fb"][col], ac["sd_global"][col])
        v = v / np.maximum(sd, 1e-6)
    elif mode == "base":
        v = v + _center_base(art, ac, col)
    return v.astype(np.float32)


# ── 사후 로짓 보정항 (post-hoc logit adjustment) ──────────────────────
# 모델 **입력**이 아니라, 예측이 끝난 로짓에 더하는 항이다.
#
#     δ(pitcher, 상황코드) = n/(n+k) · (그 셀 성공률 − 그 투수 전체 성공률)
#
# 학습에서 얼려두고(`src/ctxfit.py`), 추론에서는 행 단위 조회만 한다.
# 피처가 아니므로 모델 재학습이 필요 없고, 학습셋 타깃 리크가 구조적으로 안 생긴다.
#
# ⚠️ **새 축을 추가할 때 고칠 곳은 CTX_CODERS 하나다.**
#    §7.36에서 플래툰을 넣을 때 같은 로직이 train / verify / predict / build_submit
#    네 곳에 흩어져 있어 verify_pipeline을 빠뜨렸고, **검증이 플래툰 없는 모델을
#    통과시켰다.** 이 구조는 그 재발 경로 자체를 없애기 위한 것이다.
#
# 코더는 (정수 코드, 코드 개수 m)을 돌려준다. 키 = pitcher_id * m + code 이므로
# **m을 나중에 바꾸면 옛 아티팩트의 키가 통째로 어긋난다.** 바꾸지 말 것.


def _code_hand(df):
    """타자 손잡이. 값은 {1,2}뿐이지만 s11 아티팩트와 키를 맞추려고 m=8을 유지한다."""
    return df["batter_hand"].to_numpy(dtype=np.int64), 8


def _code_count4(df):
    """0=일반  1=3볼  2=2스트라이크  3=풀카운트급.  (train/test 모두 결측 없는 int)

    카운트는 **실패 모드의 구성**을 바꾼다 — 3볼에서는 볼넷을 피하려 존 가운데로
    몰고(실패 ①), 2스트라이크에서는 유인구로 존을 크게 벗어난다(실패 ②).
    그 트레이드오프를 얼마나 잘 관리하는지가 투수마다 다른데, `asof_*`는 이것을
    **카운트를 뭉갠 평균**으로만 담고 있다. (MODEL.md §7.37)
    """
    b = df["balls_before"].to_numpy(dtype=np.int64)
    s = df["strikes_before"].to_numpy(dtype=np.int64)
    return (b >= 3).astype(np.int64) + 2 * (s >= 2).astype(np.int64), 4


CTX_CODERS = {"hand": _code_hand, "count4": _code_count4}


def ctx_lookup(df, ax):
    """축 하나의 δ를 행 단위로 조회한다. 미지 셀은 0.

    ⚠️ **행 독립적이다.** df 전체에 대한 집계가 아니라 각 행이 자기 키로
    searchsorted 할 뿐이다 (Trackman 조회와 같은 형태).
    """
    code, m = CTX_CODERS[ax["name"]](df)
    kv = df["pitcher_id"].to_numpy(dtype=np.int64) * m + code
    keys = ax["keys"]
    j = np.clip(np.searchsorted(keys, kv), 0, len(keys) - 1)
    return np.where(keys[j] == kv, ax["delta"][j], 0.0)


# ── 시즌 진행분 (asof 차분) — MODEL.md §7.54·§7.55 ───────────────────
# `asof_*`는 커리어 누적이다. 거기서 **train에서 얼린 시즌 시작 시점 상수**를 빼면
# 그 행이 속한 시즌의 진행분이 나온다.
#
#     n_szn = asof_n − n_off(pitcher, season)          이번 시즌 던진 공
#     k_szn = asof_rate × asof_n − k_off(...)          이번 시즌 성공 횟수
#     gap   = EB(k_szn, n_szn) − 커리어 비율            "올해 − 커리어"
#
# **비율은 뺄 수 없고 횟수는 뺄 수 있다.** r×n이 정수 복원된다는 것이 전제이고
# 측정했다 (max|r×n − round| < 0.008, 저장 6자리).
#
# ⚠️ **행 독립적이다.** 각 행이 자기 (pitcher_id, season) 키로 searchsorted 할 뿐이다
#    (δ · Trackman 조회와 같은 형태). `df` 전체에 대한 집계가 없다.
#    오프셋·표준화 상수는 전부 train에서만 만들어져 artifacts에 동결된다.


def asof_lookup(df, a, with_hit=False, ct=None):
    """축 하나의 시즌 진행분 gap을 행 단위로 계산해 **표준화**해서 돌려준다.

    축은 **자기를 설명한다** — `id`/`n_col`/`r_col`이 아티팩트 안에 들어 있어서
    이 함수는 투수/타자를 구분하지 않는다. 축을 늘려도 여기는 안 고친다
    (`ctx_lookup`이 `CTX_CODERS`로 같은 일을 하는 것과 같은 구조).

    미등록(신인·데뷔 전)은 오프셋 0으로 떨어지고, 그러면 커리어 = 시즌분이 되어
    **정의상 옳다** (2024 검증에서 일치율 1.000000).

    `with_hit=True`면 **오프셋 조회 적중 마스크**를 같이 준다.
    ⚠️ 이 값을 진단으로 써야 한다 — `v != 0`을 세면 표준화 때문에 항상 100%가 나오고,
       그러면 §6 버그 2(참조행과 추론행의 적중률 불일치)를 **가린다.**
    """
    kv = (df[a["id"]].to_numpy(dtype=np.int64) * 10000
          + df["season"].to_numpy(dtype=np.int64))
    keys = a["keys"]
    j = np.clip(np.searchsorted(keys, kv), 0, max(len(keys) - 1, 0))
    hit = (len(keys) > 0) & (keys[j] == kv)

    # ⚠️ **float32로 한 번 내린 뒤 float64로 올린다** — `_col64`와 같은 이유다.
    #    학습은 train.csv를 float32 dtype으로 읽지만 평가 서버의 test.csv는 pandas
    #    기본 추론으로 float64가 된다. 곧바로 float64로 쓰면 서빙 쪽이 원본 정밀도를
    #    더 많이 갖고, `k = r × n`이 n~3000에서 1 ULP씩 어긋난다.
    #    오프셋 `k_off`는 **float32 train에서** 만들어졌으므로 뺄셈의 두 항이
    #    다른 정밀도 세계에 있게 된다.
    #    → verify_pipeline이 실제로 잡았다: max|serve − direct| = 4.527e-07 (§7.55 사고).
    f64 = lambda c: df[c].to_numpy(dtype=np.float32).astype(np.float64)
    n_now, r_car = f64(a["n_col"]), f64(a["r_col"])
    n_szn = np.maximum(n_now - np.where(hit, a["n_off"][j], 0.0), 0.0)
    k_szn = np.clip(np.nan_to_num(r_car * n_now) - np.where(hit, a["k_off"][j], 0.0),
                    0.0, None)

    # ── pre-2023 F 오염 보정 (§7.60) ────────────────────────────────
    # 이 경로는 원본 `asof_*`를 읽는다. `build_features`가 하는 보정(§7.49)이
    # 여기엔 없었고, 그건 실수였다:
    #
    #   k_szn = k_now − k_off  는 **깨끗하다.** 오염 투구는 전부 개막 이전이라
    #           두 항에 똑같이 들어가 상쇄된다 (2024·2025 투구는 새 기준).
    #   r_car 는 **오염된 채로 남는다.** 그런데 gap은 이걸 기준선으로 뺀다.
    #           → gap이 D/m 만큼 밀린다. 크기는 pre-2023 F 노출에 비례해 투수마다 다르다.
    #             2024 행 측정: 55.2%가 노출, 평균 0.87σ, 최대 10.26σ.
    #
    # `cadj = (축, 부호)`. 성공률은 과대이므로 −1, 반대성 비율은 과소이므로 +1.
    # 없으면(=None) 종전과 **비트 동일**하게 동작한다.
    cadj = a.get("cadj")
    if cadj and ct:
        import contam
        r_car = r_car + float(cadj[1]) * contam.adjust(df, ct, cadj[0])

    prior = np.where(np.isfinite(r_car), r_car, a["league"])
    s = float(a["strength"])
    gap = (k_szn + s * prior) / (n_szn + s) - prior
    v = np.nan_to_num((gap - float(a["mu"])) / float(a["sd"]))
    return (v, hit) if with_hit else v


# ── 레퍼토리 균형도 축 (MODEL.md §7.58) ────────────────────────────────────
#
# `asof_pitcher_{fastball,breaking,offspeed}_rate`를 **3그룹으로 재정규화**한 뒤
# 엔트로피를 잰다. 값이 클수록 구종을 고르게 섞는 투수다.
#
# 이 축은 앞의 둘과 **얼리는 것의 성격이 다르다.**
#   δ(ctx)      엔티티×상황 표 — 수천 셀
#   asof(gap)   오프셋 표 수천 개 + β
#   mix(r3_ent) **스칼라 6개뿐** — league 3 + mu + sd + β
# 오프셋 조회조차 없다. 제공 컬럼만의 순수 행 함수라 §7.55의 float32/float64
# 함정(k = r×n 뺄셈)이 원리상 발생하지 않는다. 그래도 dtype 규약은 맞춘다 —
# mu/sd가 float32 train에서 나왔기 때문이다.
#
# ⚠️ 배포 피처에 `pitchmix_entropy`가 **이미 있는데도** 잔차 t가 +8.72/+5.99 남았다.
#    §7.56과 같은 그림이다 — 정보가 아니라 **전달 방식**이 문제다.


def mixent_lookup(df, a, with_hit=False):
    """3그룹 재정규화 믹스율의 엔트로피를 행 단위로 계산해 표준화해서 돌려준다.

    cold(믹스 정보 없음)는 **리그 평균 믹스**로 떨어뜨린다. 0으로 두면 엔트로피가
    0이 되어 "구종 하나만 던지는 투수"와 구별이 안 된다 — 결측을 극단값으로
    바꾸는 것은 §7.49가 오염 보정에서 본 것과 같은 종류의 사고다.
    """
    f64 = lambda c: df[c].to_numpy(dtype=np.float32).astype(np.float64)
    raw = np.stack([np.nan_to_num(f64(c), nan=0.0) for c in a["cols"]], axis=1)
    tot = raw.sum(1)
    n_mix = np.nan_to_num(f64(a["n_col"]), nan=0.0)

    # `tot`는 1이 아니다 — 3그룹에 안 들어가는 구종(other)이 빠져 있다.
    # 재정규화해야 "그룹 안에서 얼마나 고른가"가 된다 (§7.58에서 `mix3_sum`,
    # 즉 other 비중 자체는 죽었다 — 재정규화가 아니라 **균형도**가 신호다).
    ok = (tot > 0) & (n_mix > 0)
    r3 = np.divide(raw, np.maximum(tot, 1e-9)[:, None])
    r3 = np.where(ok[:, None], r3, np.asarray(a["league"], dtype=np.float64)[None, :])

    q = np.clip(r3, 1e-12, 1.0)
    ent = -(q * np.log(q)).sum(1)
    v = np.nan_to_num((ent - float(a["mu"])) / float(a["sd"]))
    return (v, ok) if with_hit else v


def logit_adjust(df, art):
    """모든 보정 축의 Σ β·δ 와 축별 적중률을 돌려준다.

    학습의 참조행 · 검증 · 추론이 **전부 이 함수 하나를 통과해야 한다.**
    세 경로가 같은 코드를 보는 것이 §7.36 버그의 유일한 구조적 예방책이다.
    """
    z = np.zeros(len(df), dtype=np.float64)
    hits = {}
    for ax in ((art.get("ctx") or []) if art else []):
        d = ctx_lookup(df, ax)
        hits[ax["name"]] = float(np.mean(d != 0.0))
        z += float(ax["beta"]) * d

    # 시즌 진행분 축들 (§7.55 투수 · §7.57 타자). `ctx`와 같은 리스트 구조다 —
    # 축을 늘려도 이 루프는 안 고친다.
    for a in ((art.get("asof") or []) if art else []):
        if float(a["beta"]) == 0.0:
            continue
        v, hit = asof_lookup(df, a, with_hit=True,
                             ct=(art.get("contam") if art else None))
        hits[a["name"]] = float(np.mean(hit))    # ← 진짜 조회 적중률 (v!=0 이 아니다)
        z += float(a["beta"]) * v

    # 레퍼토리 균형도 축 (§7.58). 같은 리스트 구조 — 축을 늘려도 여기는 안 고친다.
    for a in ((art.get("mix") or []) if art else []):
        if float(a["beta"]) == 0.0:
            continue
        v, ok = mixent_lookup(df, a, with_hit=True)
        hits[a["name"]] = float(np.mean(ok))     # cold가 아닌 행의 비율
        z += float(a["beta"]) * v

    # 행 단위 game_type=F 레벨 δ_F (§7.69). game_type은 피처에서 드랍돼 있어
    # 모델이 F 행 레벨을 표현할 수 없다 — 그 몫을 사후 로짓으로 더한다.
    # 행 자신의 컬럼 하나만 보는 조회라 test 행 간 독립성에 영향이 없다.
    gt = (art.get("gt") if art else None)
    if gt and float(gt["delta"]) != 0.0:
        isf = (df["game_type"].astype("object").to_numpy() == "F")
        hits["gt_F"] = float(isf.mean())         # = F 행 비율
        z += float(gt["delta"]) * isf
    return z, hits


def fit_artifacts(train_df, weights=None, k_pitcher=200.0, k_batter=200.0, tm_table=None,
                  clip=None, ctx=None, contam_axes=None, tmb_table=None, asof=None,
                  mix=None, center_asof=False):
    """train에서만 호출. 추론 시 필요한 상수들을 계산해 딕셔너리로 반환.

    tm_table: (pitcher_id, season) → Trackman 요약 피처 DataFrame.
        `src/trackman_features.py`가 시즌 경계를 지켜 미리 만든다.
        여기서는 **순수 numpy 배열로 눌러서** 저장한다 — 추론 시 pandas 버전에
        의존하지 않고, 조회가 행 단위 searchsorted로 끝나게 하기 위함.
    """
    # clip: {컬럼: (lo, hi)} — 꼬리 winsorize 경계. `src/clipfit.fit_clip`이 학습
    # 데이터에서 뽑아 넘겨준다. 여기서는 **받아 적기만** 한다 (추론 경로에 분위수
    # 계산이 들어가면 안 되므로 — build_submit.py의 AST 스캐너가 막는다).
    art = {"priors": {}, "cat_maps": {}, "clip": dict(clip or {}),
           "ctx": list(ctx or []),      # src/ctxfit.fit_ctx 이 만든다 (사후 로짓 보정 축)
           "asof": list(asof or []),    # src/asofdelta.fit_asof_axes (시즌 진행분 축들)
           "mix": list(mix or []),      # src/mixent.fit_mixent (레퍼토리 균형도 §7.58)
           "smooth_k": {col: (k_pitcher if ncol in _PITCHER_N_COLS else k_batter)
                        for col, ncol in SMOOTH_SPEC}}

    # 투수 테이블과 타자 테이블은 조회 키의 ID 컬럼만 다르다 (§7.50).
    for slot, idcol, table in (("tm", "pitcher_id", tm_table),
                               ("tmb", "batter_id", tmb_table)):
        if table is None:
            continue
        cols = [c for c in table.columns if c not in (idcol, "season")]
        key = (table[idcol].to_numpy(dtype=np.int64) * 10000
               + table["season"].to_numpy(dtype=np.int64))
        order = np.argsort(key)
        art[slot] = {"keys": key[order],
                     "values": table[cols].to_numpy(dtype=np.float32)[order],
                     "cols": list(cols)}

    w = np.ones(len(train_df)) if weights is None else np.asarray(weights, dtype=np.float64)

    # 스무딩용 리그 평균 — 가중 평균으로 최근 시즌 쪽에 맞춘다
    for col, _ in SMOOTH_SPEC:
        v = train_df[col].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        art["priors"][col] = float(np.average(v[m], weights=w[m])) if m.any() else 0.5
    for col in PREV_COLS:
        v = train_df[col].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        art["priors"][col] = float(np.average(v[m], weights=w[m])) if m.any() else 0.5

    # 범주형 → 정수 코드. test에만 나오는 값은 -1(=결측)로 떨어진다.
    for col in CATEGORICAL_STR:
        vals = sorted(str(v) for v in train_df[col].astype("object").dropna().unique())
        art["cat_maps"][col] = {v: i for i, v in enumerate(vals)}
    for col in CATEGORICAL_INT:
        vals = sorted(int(v) for v in train_df[col].dropna().unique())
        art["cat_maps"][col] = {v: i for i, v in enumerate(vals)}

    # pre-2023 F 오염 보정표 (§7.48·§7.49). train에서만 만들어 여기에 동결한다.
    # `modelabel`은 config를 import하므로 **함수 안에서** 늦게 import한다 —
    # 제출 zip에는 features.py와 contam.py만 들어가고 이 경로는 실행되지 않는다.
    if contam_axes:
        import contam_fit as _cfit          # fit 전용 (제출물에 안 들어간다)
        from modelabel import recover as _recover
        lab, _, _ = _recover(train_df)
        rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
        art["contam"] = _cfit.fit_contam(train_df, rev)
        art["contam"]["axes"] = tuple(contam_axes)

    # (시즌 × game_type) 중심화 표 (§7.106) — **맨 마지막에, 2패스로 만든다.**
    # ⚠️ 표를 원본 컬럼에서 적합하면 안 된다 (§7.106 ⑧의 두 번째 결함). 중심화는
    #    `build_features` 안에서 **오염보정·클리핑 뒤**에 걸리는데, 오염보정은 2023·2024
    #    행에도 걸린다 — 그 투수의 **커리어에 pre-2023 F 투구가 섞여 있기 때문**이다.
    #    보정량이 누적 F 노출(≈ n)의 함수라, 원본에서 적합한 표로 빼면 잔차가 정확히
    #    **신인–베테랑 축**에 남는다. 실측에서 그 격차가 +0.008 → +0.018 로 커졌다.
    # → 중심화 없는 아티팩트로 한 번 돌려 **실제로 모델이 볼 값**에서 표를 만든다.
    if center_asof:
        mode = center_asof if isinstance(center_asof, str) else "base"
        adj = build_features(train_df, art)          # art 에 asof_center 가 아직 없다
        tmp = train_df[["season", "game_type"]].copy()
        pool = CENTER_SUBSET5 if mode in _SUBSET5_MODES else CENTER_COLS
        cols = [c for c in pool if c in adj.columns]
        for c in cols:
            tmp[c] = adj[c].to_numpy(dtype=np.float64)
        art["asof_center"] = fit_asof_center(tmp, cols=cols)
        if mode == "mu5dead":                       # §7.119 위약 — 죽은 축 그룹 평균
            tmp[_DEAD_COL] = train_df[_DEAD_COL].to_numpy()
            art["asof_center"]["dead"] = fit_dead_group(tmp, cols=cols)
        art["asof_center"]["mode"] = mode
        art["asof_center"]["season0"] = float(train_df["season"].min())
        # ★ zero 계열이면 **맞출 상수 자체를 없앤다** (§7.116).
        #   `priors` 는 스무딩 목표이자 MLP `nan_plan` 의 결측 대체값이자 `form_*` 의
        #   NaN 채움값이다. 평균 0 으로 중심화해놓고 결측을 0.52 로 채우면 그 행만
        #   10σ 밖으로 날아간다 — §7.115 ①과 같은 종류의 함정이 세 곳에 있다.
        #   여기 한 줄로 바꾸면 셋이 전부 따라온다.
        #   ⚠️ `mu5` 는 여기 없다 — 값을 안 건드리는 대조군이라 priors 가 그대로여야 한다.
        if mode in _ZERO_PRIOR_MODES:
            for c in cols:
                art["priors"][c] = 0.0
        # §7.151 — 거듭제곱 팔은 **상수도 같은 지수로 올린다.** 안 하면 스무딩이
        #   v^p 를 원래 prior(.526) 쪽으로 당기는데 v² 의 중심은 .28, v³ 은 .15 라
        #   저표본 행만 크게 끌려간다 — §7.115 결함① 과 같은 함정이다.
        #   MLP nan_plan·form_* 의 NaN 채움값도 이 한 줄을 따라온다.
        elif mode in _POW5:
            for c in cols:
                art["priors"][c] = float(art["priors"][c]) ** _POW5[mode]

    return art


def _clip32(v, art, col):
    """꼬리 winsorize. **float32 단계에서** 자른다 — `_col64`와 스무딩 경로가
    같은 숫자를 보게 하려면 자르는 시점의 dtype이 같아야 한다 (아래 skew 사고 참조).
    경계도 float32로 내려서 float64로 올릴 때 정확히 같은 값이 되게 한다."""
    b = art.get("clip", {}).get(col) if art else None
    if b is None:
        return v
    return np.clip(v, np.float32(b[0]), np.float32(b[1]))


def _col64(g, col, fill, art=None):
    """float32로 한 번 내린 뒤 float64로 올린다.

    ⚠️ 이 두 단계를 건너뛰면 train/serve skew가 난다.
    학습은 train.csv를 float32 dtype으로 읽지만, 평가 서버의 test.csv는
    pandas 기본 추론으로 **float64**가 된다. 곧바로 float64로 쓰면 서빙 쪽이
    원본 정밀도를 더 많이 갖게 되어 두 경로의 차분 피처가 1 ULP씩 어긋나고,
    LightGBM의 max_bin 경계에 걸린 행에서 예측이 갈린다.
    """
    v = _clip32(g[col].to_numpy(dtype=np.float32), art, col).astype(np.float64)
    return np.nan_to_num(v, nan=fill)


def _col64v(g, col, fill, art=None):
    """`_col64` 의 **행별 채움값** 판. `fill` 이 스칼라가 아니라 배열이라는 것만 다르다.

    dtype 왕복(float32 → float64)은 똑같이 지킨다 — 여기서 갈라지면 `_col64` 를 쓰는
    팔과 이 함수를 쓰는 팔이 1 ULP 씩 어긋나 **팔의 차이가 아니라 dtype 의 차이**를
    재게 된다 (`_col64` 주석의 skew 사고와 같은 종류).
    """
    v = _clip32(g[col].to_numpy(dtype=np.float32), art, col).astype(np.float64)
    m = np.isnan(v)
    v[m] = np.asarray(fill, dtype=np.float64)[m]
    return v


def _smooth(rate, n, prior, k):
    """경험적 베이즈: (n*rate + k*prior) / (n + k).
    n=0(결측)이면 prior로, n>>k면 원본 rate로 수렴한다."""
    rate = np.nan_to_num(rate.astype(np.float64), nan=prior)
    n = n.astype(np.float64)
    return (n * rate + k * prior) / (n + k)


def build_features(df, art):
    """행 단위 피처 생성. df 전체 통계를 절대 쓰지 않는다."""
    out = {}
    g = df

    # ── 경기 상황 (원본 통과) ──────────────────────────────
    for c in ("game_month", "game_dayofweek", "inning", "balls_before",
              "strikes_before", "outs_before", "run_top_before", "run_bot_before",
              "run_total_before", "score_diff_home", "score_diff_pitcher_team",
              "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
              "home_win_expectancy", "li", "pitcher_hand", "batter_hand"):
        out[c] = g[c].to_numpy(dtype=np.float32)

    balls = g["balls_before"].to_numpy(dtype=np.int16)
    strikes = g["strikes_before"].to_numpy(dtype=np.int16)

    # ── 카운트 상태: 0-2와 3-0은 제구 압박이 정반대다 ─────────
    out["count_state"] = (balls * 3 + strikes).astype(np.float32)   # 0..11
    out["count_diff"] = (strikes - balls).astype(np.float32)
    out["is_two_strikes"] = (strikes == 2).astype(np.float32)
    out["is_three_balls"] = (balls == 3).astype(np.float32)
    out["is_full_count"] = ((balls == 3) & (strikes == 2)).astype(np.float32)
    out["is_first_pitch"] = ((balls == 0) & (strikes == 0)).astype(np.float32)

    # ── 좌우 매치업 ────────────────────────────────────────
    ph = g["pitcher_hand"].to_numpy(dtype=np.int16)
    bh = g["batter_hand"].to_numpy(dtype=np.int16)
    out["hand_same"] = (ph == bh).astype(np.float32)
    out["hand_matchup"] = (ph * 4 + bh).astype(np.float32)

    # ── 압박 지표 ──────────────────────────────────────────
    we = g["home_win_expectancy"].to_numpy(dtype=np.float32)
    out["we_margin"] = np.abs(we - 50.0)                 # 접전일수록 0
    out["is_pitcher_home"] = (g["top_bottom"].astype("object").to_numpy() == "T").astype(np.float32)
    out["abs_score_diff"] = np.abs(g["score_diff_pitcher_team"].to_numpy(dtype=np.float32))
    out["is_late_close"] = ((g["inning"].to_numpy() >= 7) &
                            (out["abs_score_diff"] <= 2)).astype(np.float32)

    # ── asof 표본 수 ───────────────────────────────────────
    n_p = g["asof_pitcher_n"].to_numpy()
    n_b = g["asof_batter_n"].to_numpy()
    n_m = g["asof_pitcher_pitchmix_n"].to_numpy()
    out["asof_pitcher_n"] = n_p.astype(np.float32)
    out["asof_batter_n"] = n_b.astype(np.float32)
    out["asof_pitcher_pitchmix_n"] = n_m.astype(np.float32)

    # ── 결측 자체가 신호다 ─────────────────────────────────
    out["miss_pitcher_asof"] = g["asof_pitcher_success_rate"].isna().to_numpy().astype(np.float32)
    out["miss_batter_asof"] = g["asof_batter_success_rate"].isna().to_numpy().astype(np.float32)
    out["miss_prev_game"] = g["asof_pitcher_prev1_game_success_rate"].isna().to_numpy().astype(np.float32)

    # ── 경험적 베이즈 스무딩 ───────────────────────────────
    # 원본 rate(결측 유지)와 스무딩 rate를 **둘 다** 넣고 모델이 고르게 한다.
    n_map = {"asof_pitcher_n": n_p, "asof_batter_n": n_b, "asof_pitcher_pitchmix_n": n_m}

    # ── pre-2023 F 오염 보정 (§7.48·§7.49) ──────────────────
    # 여기서 고쳐야 스무딩·form_*·matchup_*·fail_share가 **전부** 보정된 값을 본다.
    # 원본 컬럼을 갈아끼우는 것이므로 삽입 위치가 곧 전파 범위다.
    # 축 목록도 artifacts에서 온다 — 이 모듈은 config를 import하지 않는다.
    cadj = {}
    ct = art.get("contam") if art else None
    if ct:
        axes = ct.get("axes", ("p", "b"))
        if "p" in axes:
            ap = contam.adjust(g, ct, "p").astype(np.float32)
            cadj["asof_pitcher_reverse_rate"] = ap    # 누락된 '반대'를 되돌려 준다
            cadj["asof_pitcher_success_rate"] = -ap   # 그만큼 성공이 아니었다
        if "b" in axes:
            cadj["asof_batter_success_rate"] = -contam.adjust(
                g, ct, "b").astype(np.float32)

    _ac = art.get("asof_center") if art else None

    # mode="zero_off" — 중심화하면서 **오프셋을 컬럼으로 되돌려준다** (§7.116 기전 검정).
    # 가설: 중심화가 지는 이유는 잡음 제거가 아니라 **복원 불가능**이다. 같은 중심화값이
    # 시즌마다 다른 원본에서 왔는데 `season` 이 피처가 아니라 모델이 구분을 못 한다.
    # `원본 = 중심화값 + μ` 이므로 μ 를 주면 복원할 수 있다 → 복구되면 가설 확인.
    # μ 표는 (시즌,gt) 조합마다 값이 전부 달라 **한 컬럼으로 조합이 유일 식별된다.**
    # ⚠️ **진단 전용이다.** 2025 의 μ 는 외삽이라 학습 범위 바깥이고, 트리는 최좌측 잎
    #    으로 보낸다 — `season` 을 피처로 넣는 것과 같은 외삽 실패다 (§2 `game_type` 전례).
    if _ac is not None and _ac.get("mode") == "zero_off":
        out["asof_center_offset"] = _center_offset(
            g, _ac, "asof_pitcher_success_rate").astype(np.float32)
        # ⚠️ 오프셋만으로는 진단이 애매하다 — F 와 R 의 레벨이 수치적으로 뒤섞인다
        #    (2024R 0.50386 이 2022F 0.50417 바로 아래에 낀다). 복구가 안 됐을 때
        #    "가설이 틀렸다" 인지 "식별자가 나빴다" 인지 못 가른다.
        #    → 시즌 인덱스도 같이 준다. **둘 다 줘도 복구가 안 되면 가설이 죽는다.**
        out["season_idx"] = (g["season"].to_numpy(dtype=np.float32)
                             - float(_ac.get("season0", 2019.0)))

    # ── §7.119 — μ 를 **컬럼으로도** 준다 (`lag5mu` 사용자 제안 / `mu5` 대조군) ──
    # 중심화는 v ← v − μ 로 정보를 **버리는** 변환이다. μ 를 같이 주면 모델이
    # v = (v−μ) + μ 로 복원할 수 있어 **가역**이 된다 — 즉 "표현을 바꾸되 정보는
    # 안 버린다". §7.116 zero_off 가 오프셋 1개 + season_idx 로 하려던 것의
    # 컬럼별 일반화판이다.
    # ⚠️ μ 는 (시즌 × game_type) 의 결정적 함수 = **사실상 시즌 지시자**다. 그래서
    #    `mu5`(중심화 없이 μ 만) 대조군이 반드시 같이 필요하다 — 안 그러면 이득/손실이
    #    중심화 때문인지 시즌 지시자 때문인지 못 가른다 (§7.116 ⑩이 빠뜨린 대조).
    # ⚠️ 검증/2025 의 μ 는 lag 규칙상 **전체 누적 상수**라 학습 최소값보다 약간 아래다.
    #    트리는 최좌측 잎으로 보내는데, 드리프트가 단조라 "최좌측 = 가장 최근 시즌"이라
    #    방향이 맞다. §7.116 의 외삽 실패(학습 범위 밖으로 크게 벗어남)와는 다르다.
    if _ac is not None and _ac.get("mode") in _MU_FEATURE_MODES:
        _dead = _ac.get("dead")               # 위약 팔이면 죽은 축 표를 대신 쓴다
        for c in _ac["lag"]:
            v_mu = (_lookup_dead(g, _dead["table"][c], _dead["global"][c])
                    if _dead is not None else
                    _lookup_group(g, _ac["lag"][c], _ac["lag_fb"][c], _ac["global"][c]))
            out[c + "_mu"] = v_mu.astype(np.float32)   # 이름을 위약과 **같게** 둔다

    sm = {}
    for col, ncol in SMOOTH_SPEC:
        # 꼬리를 먼저 자른 뒤 스무딩·파생으로 내려보낸다. 자르는 것은 값이 아니라
        # **레버리지**다 — 행은 그대로 남고 극단값만 경계에 앉는다.
        v0 = g[col].to_numpy(dtype=np.float32)
        if col in cadj:                               # NaN은 NaN으로 남는다
            v0 = np.clip(v0 + cadj[col], 0.0, 1.0)
        raw = _clip32(v0, art, col)
        # ── (시즌 × game_type) 중심화 (§7.106) ─────────────────
        # 클리핑 **뒤**여야 한다 — clip 경계는 원래 스케일에서 적합됐다.
        # 스무딩 **앞**이어야 한다 — 그래야 `_sm`·`form_*`·`matchup_*`가 전부
        # 중심화된 값을 본다 (오염 보정과 같은 이유, 삽입 위치가 전파 범위다).
        if _ac is not None and col in _ac["table"]:
            raw = _center_apply(g, art, _ac, col, raw)
        out[col] = raw                                    # 원본 (NaN 그대로 → LGBM이 처리)
        s = _smooth(raw, n_map[ncol], art["priors"][col], art["smooth_k"][col])
        sm[col] = s
        out[col + "_sm"] = s.astype(np.float32)

    for col in PREV_COLS:
        pv = _clip32(g[col].to_numpy(dtype=np.float32), art, col)
        if _ac is not None and col in _ac["table"]:
            pv = _center_apply(g, art, _ac, col, pv)
        out[col] = pv

    # ── 최근 폼 변화 (차분은 트리가 스스로 못 만든다) ────────
    pr = art["priors"]

    # ── [실험 전용] 이력 깊이 — HIST_MODE 참조 (기본값 None = 없음) ──
    # ⚠️ **원본 컬럼끼리** 비교한다. 클리핑·중심화 뒤에 재면 안 된다 — clip 은
    #    다대일 사상이라 꼬리에 앉은 서로 다른 값이 인위적으로 같아지고, 그러면
    #    "완료 경기가 적다"가 아닌 행이 얕은 깊이로 잘못 분류된다.
    d0 = d1 = d3 = None
    if HIST_MODE is not None:
        if HIST_MODE not in _HIST_MODES:
            raise ValueError(f"모르는 HIST_MODE: {HIST_MODE!r}")
        r1 = g["asof_pitcher_prev1_game_success_rate"].to_numpy(dtype=np.float64)
        r3 = g["asof_pitcher_prev3_game_success_rate"].to_numpy(dtype=np.float64)
        r5 = g["asof_pitcher_prev5_game_success_rate"].to_numpy(dtype=np.float64)
        d0 = np.isnan(r1)                            # 완료 경기 0  (= miss_prev_game)
        d1 = ~d0 & (r1 == r3) & (r3 == r5)           #            1
        d3 = ~d0 & ~d1 & (r3 == r5)                  #            2~3
        depth = np.where(d0, 0.0, np.where(d1, 1.0, np.where(d3, 2.0, 3.0)))
        # ⚠️ `carry` 팔에는 **넣지 않는다.** carry 는 (채움방식 x 플래그) 2x2 의 한 칸이라
        #    컬럼이 하나 더 붙으면 한 팔에 변경이 둘이 되어 칸의 의미가 깨진다.
        if HIST_MODE in ("depth", "depthnan"):
            out["hist_depth"] = depth.astype(np.float32)
        elif HIST_MODE == "depthflag":
            # 기존 키를 **덮어쓴다** — dict 는 재대입해도 삽입 순서를 유지하므로
            # 컬럼 수도 순서도 그대로다. 이게 이 팔의 전부이자 존재 이유다.
            assert "miss_prev_game" in out, "플래그가 아직 안 만들어졌다 (삽입 순서 확인)"
            out["miss_prev_game"] = ((3.0 - depth) / 3.0).astype(np.float32)

    if HIST_MODE == "carry":
        # 결측을 리그 prior 가 아니라 **그 투수 자신의 커리어(스무딩)** 로 채운다.
        # 그러면 form_* 이 전부 정확히 0 이 되어, 현행의 `prior - career_i`
        # (커리어에 대한 가짜 단조감소 기울기)가 사라진다. 레벨은 플래그가 흡수한다.
        p1 = _col64v(g, "asof_pitcher_prev1_game_success_rate", sm["asof_pitcher_success_rate"], art)
        p3 = _col64v(g, "asof_pitcher_prev3_game_success_rate", sm["asof_pitcher_success_rate"], art)
        p5 = _col64v(g, "asof_pitcher_prev5_game_success_rate", sm["asof_pitcher_success_rate"], art)
        m1 = _col64v(g, "asof_pitcher_prev1_game_middle_rate", sm["asof_pitcher_middle_rate"], art)
    else:
        p1 = _col64(g, "asof_pitcher_prev1_game_success_rate", pr["asof_pitcher_prev1_game_success_rate"], art)
        p3 = _col64(g, "asof_pitcher_prev3_game_success_rate", pr["asof_pitcher_prev3_game_success_rate"], art)
        p5 = _col64(g, "asof_pitcher_prev5_game_success_rate", pr["asof_pitcher_prev5_game_success_rate"], art)
        m1 = _col64(g, "asof_pitcher_prev1_game_middle_rate", pr["asof_pitcher_prev1_game_middle_rate"], art)

    career = sm["asof_pitcher_success_rate"]
    out["form_p1_minus_career"] = (p1 - career).astype(np.float32)
    out["form_p3_minus_career"] = (p3 - career).astype(np.float32)
    out["form_p1_minus_p5"] = (p1 - p5).astype(np.float32)
    out["form_p3_minus_p5"] = (p3 - p5).astype(np.float32)
    out["form_mid_p1_minus_career"] = (m1 - sm["asof_pitcher_middle_rate"]).astype(np.float32)

    if HIST_MODE == "depthnan":
        # 구조적 0 → NaN. p1==p5 는 깊이 <=1, p3==p5 는 깊이 <=3 에서 성립한다.
        # 깊이 0 은 **정확히 0 이 아니라** prior 끼리의 차인 상수(-0.0026 / -0.0013)로
        # 앉는다 — 컬럼마다 prior 가 다르기 때문이다. 값은 달라도 "그 행에 대해
        # 아무것도 말해주지 않는 고정값"이라는 성질은 같으므로 함께 지운다.
        out["form_p1_minus_p5"] = np.where(d0 | d1, np.nan,
                                           out["form_p1_minus_p5"]).astype(np.float32)
        out["form_p3_minus_p5"] = np.where(d0 | d1 | d3, np.nan,
                                           out["form_p3_minus_p5"]).astype(np.float32)

    # ── 매치업 엣지: 투수 제구력 vs 타자가 겪어온 제구력 ─────
    out["matchup_success_edge"] = (career - sm["asof_batter_success_rate"]).astype(np.float32)
    out["matchup_middle_edge"] = (sm["asof_pitcher_middle_rate"]
                                  - sm["asof_batter_middle_rate"]).astype(np.float32)

    # ── 실패 3유형의 구성비 (몰림 vs 반대방향) ──────────────
    fail = sm["asof_pitcher_middle_rate"] + sm["asof_pitcher_reverse_rate"]
    out["fail_middle_share"] = (sm["asof_pitcher_middle_rate"] /
                                np.maximum(fail, 1e-6)).astype(np.float32)
    out["pitcher_ball_minus_strike"] = (sm["asof_pitcher_ball_rate"]
                                        - sm["asof_pitcher_strike_rate"]).astype(np.float32)

    # ── [실험 전용] 곱 상호작용 — EXTRA_PROD 참조 (기본값 None = 없음) ──
    if EXTRA_PROD == "prod":
        out["revxsuc"] = (out["asof_pitcher_reverse_rate"]
                          * out["asof_pitcher_success_rate"]).astype(np.float32)
    elif EXTRA_PROD == "placebo":
        dow = g["game_dayofweek"].to_numpy(dtype=np.float32) / 6.0
        dead = _DOW_LO + (_DOW_HI - _DOW_LO) * dow
        out["revxsuc"] = (dead * out["asof_pitcher_success_rate"]).astype(np.float32)
    elif EXTRA_PROD is not None:
        raise ValueError(f"모르는 EXTRA_PROD: {EXTRA_PROD!r}")

    # ── 구종 레퍼토리 다양성 (엔트로피) ─────────────────────
    mix = np.stack([sm["asof_pitcher_fastball_rate"],
                    sm["asof_pitcher_breaking_rate"],
                    sm["asof_pitcher_offspeed_rate"]], axis=1)
    mix = np.clip(mix, 1e-6, 1.0)
    mix = mix / mix.sum(axis=1, keepdims=True)
    out["pitchmix_entropy"] = (-(mix * np.log(mix)).sum(axis=1)).astype(np.float32)
    out["pitchmix_nonfastball"] = (1.0 - mix[:, 0]).astype(np.float32)

    # ── Trackman 투수 요약 (시즌 경계 지킨 사전계산 테이블 조회) ──
    # 각 행이 자기 (pitcher_id, season) 키로 조회할 뿐이다 —
    # df 전체에 대한 집계가 아니므로 행 독립성이 유지된다.
    # `season`은 조회 키로만 쓰고 모델 피처로는 내보내지 않는다.
    # 투수 테이블("tm", 키 = pitcher_id)과 타자 테이블("tmb", 키 = batter_id)이
    # 같은 조회 코드를 쓴다 — §7.50의 타자 매핑으로 후자가 생겼다.
    for slot, idcol, flag in (("tm", "pitcher_id", "tm_matched"),
                              ("tmb", "batter_id", "tmb_matched")):
        if slot not in art:
            continue
        tm = art[slot]
        k = (g[idcol].to_numpy(dtype=np.int64) * 10000
             + g["season"].to_numpy(dtype=np.int64))
        pos = np.searchsorted(tm["keys"], k)
        pos_c = np.clip(pos, 0, len(tm["keys"]) - 1)
        hit = tm["keys"][pos_c] == k
        vals = tm["values"][pos_c]
        for j, c in enumerate(tm["cols"]):
            out[c] = np.where(hit, vals[:, j], np.nan).astype(np.float32)
        out[flag] = hit.astype(np.float32)

    # ── 범주형 → 정수 코드 (미지값 -1) ──────────────────────
    for col in CATEGORICAL_STR:
        codes = g[col].astype("object").astype(str).map(art["cat_maps"][col])
        out[col] = codes.fillna(-1).to_numpy(dtype=np.float64).astype(np.float32)
    for col in CATEGORICAL_INT:
        codes = g[col].astype("int64").map(art["cat_maps"][col])
        out[col] = codes.fillna(-1).to_numpy(dtype=np.float64).astype(np.float32)

    X = pd.DataFrame(out)
    return X


__all__ = ["fit_artifacts", "build_features", "logit_adjust", "ctx_lookup",
           "asof_lookup", "CTX_CODERS", "CATEGORICAL_COLS"]
