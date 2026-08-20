"""★ 구종 예측 축 — 관문 1: 잔차 t + 셔플 대조군 (§7.58 후보)

주장 (사용자)
    정보량으로는 기존 피처의 함수지만, **구종 라벨로 직접 감독**하면
    control_success 감독만으로는 못 찾는 함수를 증류해 넘겨줄 수 있다.
    δ(hand)·δ(count4)도 기존 컬럼의 결정적 함수였는데 LB를 올렸다 — 같은 논리.

죽은 운반체를 피하는 설계 (§7.39·§7.51 교훈)
    예측기에 투수 ID를 넣지 않는다. 투수 정체성은 **as-of 믹스율 3개로만** 들어간다.
    → 전이되는 것은 전역 함수 f(카운트, 좌우, 이닝, 믹스율)이고, 입력은 2025 행에서
      행마다 갱신된다. "2024에 얼린 투수별 상수"가 아니다.
    → 부산물: ID 매핑이 아예 필요 없다. trackman 안에서 as-of 믹스율을 자체 계산해
      학습하고, main 행에는 제공된 `asof_pitcher_*_rate`를 꽂는다 (3그룹 재정규화로
      분모 차이를 지운다).

무엇이 새 정보인가 — 성분 분해
    ① 투수 기저 믹스          → 이미 피처 (asof_pitcher_*_rate)
    ② 리그 공통 카운트 반응    → lg_* (no-mix 모델)로 분리 측정. 카운트 피처가
                                이미 나르고 있어야 정상 (t≈0 기대)
    ③ 믹스율×상황 상호작용     → **ptdev_* = 예측 − 커리어 믹스율**. 헤드라인.
                                (주효과를 빼고 시작 — 메모리 "잔차에 먼저 물어라")

관문
    resid_t(배포 전체 잔차, 피처) — 양 fold 부호 일치 + |t| > 2|대조군 t|.
    대조군 = fold 안에서 믹스 입력 4열(r3 + log n)을 행 셔플 후 재예측.
    t가 커도 채택 근거가 아니다 (§7.9) — β 스윕/컬럼은 다음 단계.

실행:  .venv/Scripts/python.exe -u src/pitchtype.py 2>&1 | tee experiments/pitchtype.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd

import asofdelta
import ctxfit
import foldcache
from config import (ASOF_AXES, ASOF_STRENGTH, CTX_AXES, CTX_K, DATA_DIR,
                    MLP_ALPHA, SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

TM_CSV = DATA_DIR / "trackman_history.csv"
FOLDS = [2023, 2024]
GROUPS = ["fastball", "breaking", "offspeed"]
MIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
       "asof_pitcher_offspeed_rate"]

PT_PARAMS = dict(
    objective="multiclass", num_class=3, learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=1000, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=1, max_bin=127,
    num_threads=6, verbosity=-1, seed=SEED,
)
PT_ROUNDS = 250


def _hand01(s):
    """좌우 코드 정규화 → 좌=1. 어휘가 테이블마다 다르다: tm 'Left'/'Right',
    main '1'/'2'. 빈도로 정렬해 확정했다 (투수 좌 비율 tm .2509 ↔ main '1' .2585,
    타자 tm .4604 ↔ '1' .4717) — '1'=Left, '2'=Right."""
    c = s.astype(str).str.strip().str.upper().str[:1]
    return c.isin(["L", "1"]).to_numpy(np.float64)


def load_trackman():
    """trackman 전체 + 행 단위 as-of 커리어 믹스(3그룹 재정규화).

    현재 투구를 **뺀** 직전까지의 누적이다 (groupby cumsum − 현재 행 one-hot).
    'other'는 라벨에서도 분모에서도 뺀다 — main 쪽을 3그룹 재정규화로 맞추므로.
    """
    tm = pd.read_csv(TM_CSV, usecols=[
        "season", "game_date", "trackman_game_id", "pitch_no", "inning",
        "balls_before", "strikes_before", "outs_before",
        "pitcher_trackman_id", "pitcher_hand", "batter_hand", "pitch_type_group"])
    tm = tm.sort_values(["game_date", "trackman_game_id", "pitch_no"],
                        kind="stable").reset_index(drop=True)

    oh = np.stack([(tm["pitch_type_group"] == g).to_numpy(np.float64)
                   for g in GROUPS], axis=1)
    cum = (pd.DataFrame(oh, columns=GROUPS)
           .groupby(tm["pitcher_trackman_id"].to_numpy(), sort=False)
           .cumsum().to_numpy()) - oh
    n3 = cum.sum(1)
    tm["n3"] = n3
    for i, g in enumerate(GROUPS):
        tm["r3_" + g] = np.divide(cum[:, i], np.maximum(n3, 1.0))
    tm["y"] = -1
    for i, g in enumerate(GROUPS):
        tm.loc[tm["pitch_type_group"] == g, "y"] = i
    return tm


def tm_matrix(tm, league):
    """공용 피처 행렬. cold-start(n3=0)는 리그 믹스로 대체 — main 쪽과 같은 규칙."""
    cold = tm["n3"].to_numpy() <= 0
    r3 = np.stack([tm["r3_" + g].to_numpy(np.float64) for g in GROUPS], axis=1)
    r3[cold] = league
    x = np.column_stack([
        tm["balls_before"].to_numpy(np.float64),
        tm["strikes_before"].to_numpy(np.float64),
        tm["outs_before"].to_numpy(np.float64),
        np.clip(tm["inning"].to_numpy(np.float64), 1, 12),
        _hand01(tm["pitcher_hand"]),
        _hand01(tm["batter_hand"]),
        r3,
        np.log1p(np.maximum(tm["n3"].to_numpy(np.float64), 0.0)),
    ])
    return x


FEAT_NAMES = ["balls", "strikes", "outs", "inning", "p_left", "b_left",
              "r3_ff", "r3_br", "r3_os", "log_n"]
NOMIX = [0, 1, 2, 3, 4, 5]          # 믹스 입력을 뺀 대조 모델의 열 인덱스

# 관문 2 대상: 관문 1 생존자(1차 실행) + 주효과 대조. 부호는 스윕이 알아서 가른다.
SWEEP = ["pt_ff", "pt_os", "pt_ent", "r3_ff", "r3_os", "r3_ent", "mix3_sum"]
BETAS = [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02, 0.04]


def main_matrix(d, league):
    """main 행 → 같은 스키마. 제공 asof 믹스율을 3그룹 재정규화해 꽂는다."""
    raw = np.stack([np.nan_to_num(d[c].to_numpy(np.float64), nan=0.0)
                    for c in MIX], axis=1)
    tot = raw.sum(1)
    n_mix = np.nan_to_num(d["asof_pitcher_pitchmix_n"].to_numpy(np.float64), nan=0.0)
    cold = (tot <= 0) | (n_mix <= 0)
    r3 = np.divide(raw, np.maximum(tot, 1e-9)[:, None])
    r3[cold] = league
    n3 = np.where(cold, 0.0, n_mix * tot)       # 3그룹 커리어 투구 수 복원
    x = np.column_stack([
        d["balls_before"].to_numpy(np.float64),
        d["strikes_before"].to_numpy(np.float64),
        d["outs_before"].to_numpy(np.float64),
        np.clip(d["inning"].to_numpy(np.float64), 1, 12),
        _hand01(d["pitcher_hand"]),
        _hand01(d["batter_hand"]),
        r3,
        np.log1p(n3),
    ])
    return x, r3, tot


def mlogloss(y, p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(1)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)

    tm = load_trackman()
    ok = tm["y"].to_numpy() >= 0
    print(f"trackman {len(tm):,}행  3그룹 {ok.sum():,}행 "
          f"({ok.mean():.3f})  구성 "
          + " ".join(f"{g} {float((tm['y'] == i).mean()):.3f}"
                     for i, g in enumerate(GROUPS)))
    print(f"손잡이 어휘  tm p:{sorted(tm['pitcher_hand'].astype(str).unique())} "
          f"b:{sorted(tm['batter_hand'].astype(str).unique())}  "
          f"main p:{sorted(df['pitcher_hand'].astype(str).unique())} "
          f"b:{sorted(df['batter_hand'].astype(str).unique())}")
    print(f"기준선 = 배포 전체 (블렌드 α={MLP_ALPHA} + δ{CTX_AXES} + asof {ASOF_AXES})\n")

    keep, sweeps = {}, {}
    for fold in FOLDS:
        # ── ① 구종 예측기: trackman season < fold 로 학습 ──────────────
        fit = ok & (tm["season"].to_numpy() < fold)
        hld = ok & (tm["season"].to_numpy() == fold)
        yf = tm.loc[fit, "y"].to_numpy()
        league = np.array([float((yf == i).mean()) for i in range(3)])
        xf = tm_matrix(tm[fit], league)

        m_full = lgb.train(PT_PARAMS, lgb.Dataset(xf, label=yf), PT_ROUNDS)
        m_nomix = lgb.train(PT_PARAMS, lgb.Dataset(xf[:, NOMIX], label=yf), PT_ROUNDS)

        # ── ② 원천에서의 증류량 — trackman holdout(=fold 시즌) logloss ──
        yh = tm.loc[hld, "y"].to_numpy()
        xh = tm_matrix(tm[hld], league)
        r3h = xh[:, 6:9]
        ll = dict(
            const=mlogloss(yh, np.tile(league, (len(yh), 1))),
            rates=mlogloss(yh, r3h / r3h.sum(1, keepdims=True)),
            nomix=mlogloss(yh, m_nomix.predict(xh[:, NOMIX])),
            full=mlogloss(yh, m_full.predict(xh)),
        )
        print("=" * 78)
        print(f"fold {fold}  구종 예측기 (tm fit {fit.sum():,} → holdout {hld.sum():,})")
        print(f"  logloss  상수 {ll['const']:.4f}  믹스율 {ll['rates']:.4f}  "
              f"no-mix {ll['nomix']:.4f}  full {ll['full']:.4f}")
        print(f"  증류량   상황만 {ll['const'] - ll['nomix']:+.4f}  "
              f"믹스만 {ll['const'] - ll['rates']:+.4f}  "
              f"★상호작용(full−믹스율) {ll['rates'] - ll['full']:+.4f}")

        # ── ③ 배포 전체 상태의 잔차 (mixresid.incremental과 동일) ──────
        tr, va = s < fold, s == fold
        d = df[va].copy()
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        ctx = ctxfit.fit_ctx(df[tr], TARGET_COL, CTX_AXES, CTX_K)
        a = asofdelta.fit_asof_axes(df[tr], ASOF_AXES, ASOF_STRENGTH)
        adj, _ = logit_adjust(d, {"ctx": ctx, "asof": a})
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
        r = yv - p0

        # ── ④ main 행 예측 → 피처 ─────────────────────────────────────
        xm, r3m, tot = main_matrix(d, league)
        pt = m_full.predict(xm)
        lgp = m_nomix.predict(xm[:, NOMIX])
        feat = {}
        for i, nm in enumerate(("ff", "br", "os")):
            feat["pt_" + nm] = pt[:, i]
            feat["ptdev_" + nm] = pt[:, i] - r3m[:, i]
            feat["lg_" + nm] = lgp[:, i]
        feat["pt_ent"] = entropy(pt)
        # 주효과 대조 — 예측기 없이 되는 변환들. pt_*의 t가 이것들과 같으면
        # 증류의 실체는 "재정규화된 rate"이지 상호작용이 아니다.
        for i, nm in enumerate(("ff", "br", "os")):
            feat["r3_" + nm] = r3m[:, i]
        feat["r3_ent"] = entropy(r3m)
        feat["mix3_sum"] = np.nan_to_num(tot, nan=0.0)   # = 1 − other 비중

        # 대조군 — fold 안에서 믹스 입력 4열만 행 셔플 후 재예측.
        # 카운트·좌우는 그대로 두므로 ② 성분은 남고 ③ 연결만 끊긴다.
        perm = rng.permutation(len(xm))
        xs = xm.copy()
        xs[:, 6:10] = xm[perm][:, 6:10]
        pts = m_full.predict(xs)
        fsh = {}
        for i, nm in enumerate(("ff", "br", "os")):
            fsh["pt_" + nm] = pts[:, i]
            fsh["ptdev_" + nm] = pts[:, i] - r3m[perm][:, i]
            fsh["lg_" + nm] = feat["lg_" + nm]      # 믹스 무관 — 대조 정의상 동일
        fsh["pt_ent"] = entropy(pts)
        for i, nm in enumerate(("ff", "br", "os")):
            fsh["r3_" + nm] = r3m[perm][:, i]
        fsh["r3_ent"] = entropy(r3m[perm])
        fsh["mix3_sum"] = feat["mix3_sum"][perm]

        print(f"\nfold {fold}  기준 BSS {bss(yv, p0):.1f}  행 {len(d):,}")
        print(f"  {'피처':<12}{'잔차 t':>10}{'대조군 t':>11}{'차':>9}")
        for k in feat:
            _, t = resid_t(r, feat[k])
            _, tc = resid_t(r, fsh[k])
            keep.setdefault(k, []).append((t, tc))
            mark = "  ✓" if abs(t) > 2 and abs(t) > 2 * abs(tc) else ""
            print(f"  {k:<12}{t:>10.2f}{tc:>11.2f}{t - tc:>9.2f}{mark}")

        # ── ⑤ 관문 2: 사후 로짓 β 스윕 (학습 없음, §7.37·§7.55와 같은 전달) ──
        base = bss(yv, p0)
        print(f"\n  β 스윕 (표준화 피처, ΔBSS vs {base:.1f})")
        hdr = "  " + f"{'피처':<12}" + "".join(f"{b:>8.3f}" for b in BETAS)
        print(hdr)
        for k in SWEEP:
            x = feat[k]
            xc = (x - x.mean()) / max(float(x.std()), 1e-9)
            row = []
            for b in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj + b * xc))),
                                   yv.mean())
                row.append(bss(yv, p) - base)
            sweeps.setdefault(k, []).append(row)
            print("  " + f"{k:<12}" + "".join(f"{v:>8.2f}" for v in row))

    print("\n" + "=" * 78)
    print("판정 — 부호 일치 + 대조군 초과   (lg_*는 성분 ②의 분리 측정치)")
    print("=" * 78)
    print(f"  {'피처':<12}{'t(2023)':>10}{'t(2024)':>10}{'부호일치':>10}{'대조군초과':>12}")
    for k, v in keep.items():
        (t1, c1), (t2, c2) = v
        same = (t1 > 0) == (t2 > 0)
        over = abs(t1) > 2 * abs(c1) and abs(t2) > 2 * abs(c2)
        print(f"  {k:<12}{t1:>10.2f}{t2:>10.2f}{'✓' if same else '✗':>10}"
              f"{'✓' if over else '✗':>12}")
    print("\n  읽는 법")
    print("  · ptdev_* 가 헤드라인 — 살면 '증류된 상호작용이 잔차에 남아 있다'")
    print("  · lg_* 만 살면 리그 카운트 구조를 GBDT가 안 뽑았다는 뜻 (다른 처방)")
    print("  · 전부 죽으면 축을 닫는다 — 원천 증류량(②)과 함께 §7.58에 기록")


if __name__ == "__main__":
    main()
