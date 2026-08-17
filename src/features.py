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

# 범주형 → 정수 코드 (미지값 -1).
# str/int을 분리해 두는 이유: train은 dtype을 명시해 읽지만 평가 서버의 test는
# pandas 기본 추론으로 읽힐 수 있다. int16 vs int64가 섞이면 dict 조회가 조용히
# 전부 실패해 -1로 떨어진다 — 에러 없이 성능만 죽는 종류의 버그다.
CATEGORICAL_STR = ["base_state", "top_bottom", "game_type"]
CATEGORICAL_INT = ["pitcher_team_id", "batter_team_id"]
CATEGORICAL_COLS = CATEGORICAL_STR + CATEGORICAL_INT

# prev{1,3,5}_game_* 는 항상 같이 결측된다 (등판 이력 없음) → 플래그 하나로 충분
PREV_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]


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


def asof_lookup(df, a, with_hit=False):
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
        v, hit = asof_lookup(df, a, with_hit=True)
        hits[a["name"]] = float(np.mean(hit))    # ← 진짜 조회 적중률 (v!=0 이 아니다)
        z += float(a["beta"]) * v

    # 레퍼토리 균형도 축 (§7.58). 같은 리스트 구조 — 축을 늘려도 여기는 안 고친다.
    for a in ((art.get("mix") or []) if art else []):
        if float(a["beta"]) == 0.0:
            continue
        v, ok = mixent_lookup(df, a, with_hit=True)
        hits[a["name"]] = float(np.mean(ok))     # cold가 아닌 행의 비율
        z += float(a["beta"]) * v
    return z, hits


def fit_artifacts(train_df, weights=None, k_pitcher=200.0, k_batter=200.0, tm_table=None,
                  clip=None, ctx=None, contam_axes=None, tmb_table=None, asof=None,
                  mix=None):
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

    sm = {}
    for col, ncol in SMOOTH_SPEC:
        # 꼬리를 먼저 자른 뒤 스무딩·파생으로 내려보낸다. 자르는 것은 값이 아니라
        # **레버리지**다 — 행은 그대로 남고 극단값만 경계에 앉는다.
        v0 = g[col].to_numpy(dtype=np.float32)
        if col in cadj:                               # NaN은 NaN으로 남는다
            v0 = np.clip(v0 + cadj[col], 0.0, 1.0)
        raw = _clip32(v0, art, col)
        out[col] = raw                                    # 원본 (NaN 그대로 → LGBM이 처리)
        s = _smooth(raw, n_map[ncol], art["priors"][col], art["smooth_k"][col])
        sm[col] = s
        out[col + "_sm"] = s.astype(np.float32)

    for col in PREV_COLS:
        out[col] = _clip32(g[col].to_numpy(dtype=np.float32), art, col)

    # ── 최근 폼 변화 (차분은 트리가 스스로 못 만든다) ────────
    pr = art["priors"]
    p1 = _col64(g, "asof_pitcher_prev1_game_success_rate", pr["asof_pitcher_prev1_game_success_rate"], art)
    p3 = _col64(g, "asof_pitcher_prev3_game_success_rate", pr["asof_pitcher_prev3_game_success_rate"], art)
    p5 = _col64(g, "asof_pitcher_prev5_game_success_rate", pr["asof_pitcher_prev5_game_success_rate"], art)
    career = sm["asof_pitcher_success_rate"]
    out["form_p1_minus_career"] = (p1 - career).astype(np.float32)
    out["form_p3_minus_career"] = (p3 - career).astype(np.float32)
    out["form_p1_minus_p5"] = (p1 - p5).astype(np.float32)
    out["form_p3_minus_p5"] = (p3 - p5).astype(np.float32)

    m1 = _col64(g, "asof_pitcher_prev1_game_middle_rate", pr["asof_pitcher_prev1_game_middle_rate"], art)
    out["form_mid_p1_minus_career"] = (m1 - sm["asof_pitcher_middle_rate"]).astype(np.float32)

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
