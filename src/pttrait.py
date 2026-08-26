"""§7.170 — 사용자 제안: 구종 예측 × 투수별 구종별 물리량(구속·무브먼트·릴리즈)

제안: trackman에서 (투수, 구종)별 물리량 통계를 뽑아, 구종 예측 확률과 결합해
"다음 투구의 기대 물리량"을 피처로 넣으면 어떤가.

성분 분해 — 무엇이 새 것인가
    E_X  = Σ_t p_t·X[p,t]           (기대 물리량)
    lvl_X = Σ_t r3_t·X[p,t]         (커리어 믹스 가중 평균 = 주효과. §7.113~116에서
                                     죽은 "동결 투수 상수" 가계 — 대조군으로만)
    dev_X = E_X − lvl_X = Σ_t (p_t−r3_t)·X[p,t]   ← ★ 헤드라인. 상황이 만드는
                                     기대 물리량의 변화. 행마다 변한다.

E1 형질성 사전 검증: 쓰는 통계 전부 mean류 (rel_speed·ivb·hb·relh·rels·ext 평균,
인접 시즌 상관 +0.93~0.98) — "얼린 상수" 관문은 통과 상태에서 출발.

위약 2종 (메모리: 위약은 공유 항을 남겨야 한다 / 정체만 지운 위약)
    LEAG  X[p,t] → 리그 X[t]  (p−r3 기하·물리량 스케일 유지, 투수 정체만 삭제)
    PERM  매핑된 투수끼리 X 벡터를 순열 (주변 분포 유지, 정체 삭제)

관문: A층(블렌드+δ, fold 2023·2024) + B층(s40 배포층, fold 2024) resid_t.
    양 fold 부호 일치 + |t|>2 + |t| > 2·max(|t_LEAG|, |t_PERM|).

실행:  .venv/Scripts/python.exe -u src/pttrait.py 2>&1 | tee experiments/pttrait.log
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

import foldcache
import jointbeta as J
import pitchtype as PT
from config import (ASOF_AXES, CAT_ALPHA, CTX_AXES, FSPEC_ALPHA, MLP_ALPHA,
                    SEED, TARGET_COL, TRAIN_CSV)
from ctxsplit import resid_t
from data import load_csv
from hbeta2 import train_fspec
from predict import PROB_CENTER, PROB_SCALE
from smotenc import prep
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
MAP_CSV = Path(__file__).parent.parent / "model_s28" / "pitcher_id_map.csv"
TRAITS = ["rel_speed", "induced_vert_break", "horz_break",
          "rel_height", "rel_side", "extension"]
MIN_CELL = 50        # (투수,구종) 셀 노이즈 게이트 (메모리: 비율 조건에는 노이즈 게이트)
BETAS = [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]


def trait_tables(tm_full, fold):
    """season<fold 로 얼린 (투수,구종)별 평균 + 리그 구종별 평균."""
    m = (tm_full["season"].to_numpy() < fold) & (tm_full["y"].to_numpy() >= 0)
    sub = tm_full[m]
    g = sub.groupby(["pitcher_trackman_id", "y"])
    agg = g[TRAITS].mean()
    n = g.size().rename("n")
    agg = agg[n >= MIN_CELL]
    league = sub.groupby("y")[TRAITS].mean()          # 3×6
    return agg, league, int((n >= MIN_CELL).sum()), len(n)


def build_dev(d, pt, r3m, agg, league, pid2tid, rng):
    """행별 dev/lvl 행렬 + 위약 2종. 반환: dict[name] -> (n_rows,) 벡터."""
    tid = d["pitcher_id"].map(pid2tid)
    # 투수별 3×6 물리량 행렬. 셀 결측은 리그 구종 평균으로.
    L = league.to_numpy()                              # (3, 6)
    tids = [t for t in tid.dropna().unique() if (t in agg.index.get_level_values(0))]
    mats = {}
    for t in tids:
        M = L.copy()
        sub = agg.loc[t]
        for yy in sub.index:
            M[int(yy)] = sub.loc[yy].to_numpy()
        mats[t] = np.where(np.isnan(M), L, M)   # 측정 결측 셀은 리그로
    # 순열 위약: 매핑된 투수끼리 행렬을 섞는다
    perm_keys = list(mats)
    perm_vals = [mats[k] for k in perm_keys]
    order = rng.permutation(len(perm_keys))
    mats_perm = {k: perm_vals[j] for k, j in zip(perm_keys, order)}

    n = len(d)
    stackM = np.tile(L, (n, 1, 1))                     # 기본 = 리그 (미매핑·미보유)
    stackP = stackM.copy()
    covered = np.zeros(n, bool)
    tv = tid.to_numpy()
    for i in range(n):
        t = tv[i]
        if t == t and t in mats:                       # NaN 체크
            stackM[i] = mats[t]
            stackP[i] = mats_perm[t]
            covered[i] = True

    w_dev = (pt - r3m)                                 # (n, 3)  Σ=0
    out = {}
    for j, x in enumerate(TRAITS):
        out["dev_" + x] = np.einsum("nt,nt->n", w_dev, stackM[:, :, j])
        out["lvl_" + x] = np.einsum("nt,nt->n", r3m, stackM[:, :, j])
        out["devLEAG_" + x] = w_dev @ L[:, j]
        out["devPERM_" + x] = np.einsum("nt,nt->n", w_dev, stackP[:, :, j])
    out = {k: np.nan_to_num(v, nan=0.0) for k, v in out.items()}
    assert all(np.isfinite(v).all() for v in out.values())
    return out, covered


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    rng = np.random.default_rng(SEED)
    pid_map = pd.read_csv(MAP_CSV)
    pid2tid = dict(zip(pid_map["pitcher_id"], pid_map["pitcher_trackman_id"]))
    print(f"ID 매핑 {len(pid2tid)}명   형질: {TRAITS}   셀 게이트 n≥{MIN_CELL}")

    tm = PT.load_trackman()
    # PT.load_trackman은 측정 컬럼을 안 읽는다 — 같은 키로 같은 정렬을 밟아 붙인다
    raw = pd.read_csv(PT.TM_CSV, usecols=["game_date", "trackman_game_id",
                                          "pitch_no"] + TRAITS)
    raw = raw.sort_values(["game_date", "trackman_game_id", "pitch_no"],
                          kind="stable").reset_index(drop=True)
    for c in TRAITS:
        tm[c] = raw[c].to_numpy()
    ok = tm["y"].to_numpy() >= 0

    keep = {}
    for fold in FOLDS:
        fit = ok & (tm["season"].to_numpy() < fold)
        yf = tm.loc[fit, "y"].to_numpy()
        league_mix = np.array([float((yf == i).mean()) for i in range(3)])
        xf = PT.tm_matrix(tm[fit], league_mix)
        m_full = lgb.train(PT.PT_PARAMS, lgb.Dataset(xf, label=yf), PT.PT_ROUNDS)

        agg, league_tr, ncell, ntot = trait_tables(tm, fold)
        va_m = s == fold
        d = df[va_m].copy()
        yv = cached[fold][2]
        xm, r3m, _ = PT.main_matrix(d, league_mix)
        pt = m_full.predict(xm)
        feat, covered = build_dev(d, pt, r3m, agg, league_tr, pid2tid, rng)

        print("\n" + "=" * 78)
        print(f"fold {fold}  셀 통과 {ncell}/{ntot}  행 커버리지 {covered.mean():.3f}")

        # A층
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        zA = z + Xax @ dep

        def probA(zz, m=yv.mean()):
            return shift_to_prior(1.0 / (1.0 + np.exp(-zz)), m)

        rA = yv - probA(zA)
        layers = [("A", rA, zA, probA)]

        # B층 (fold 2024)
        if fold == 2024:
            zl = np.load(CACHE / "sfull_f2024_SMOTE_lgb.npz")["z"]
            cb = np.load(CACHE / "sfull_f2024_base_cat.npz")["z"]
            mb = np.mean([np.load(p)["z"] for p in
                          sorted(CACHE.glob("sfull_f2024_base_mlp_*.npz"))], axis=0)
            tr2, va2, w2, Xfn, art2 = prep(df, fold)
            z_fs, _ = train_fspec(tr2, Xfn, Xfn(va2))
            isF = va2["game_type"].to_numpy().astype(str) == "F"
            zB = (1 - CAT_ALPHA) * ((1 - MLP_ALPHA) * zl + MLP_ALPHA * mb) \
                + CAT_ALPHA * cb
            zB = zB.copy()
            zB[isF] = (1 - FSPEC_ALPHA) * zB[isF] + FSPEC_ALPHA * z_fs[isF]
            zB = zB + Xax @ dep

            def probB(zz, m=yv.mean()):
                p = shift_to_prior(1.0 / (1.0 + np.exp(-zz)), m)
                return shift_to_prior(
                    PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), m)

            rB = yv - probB(zB)
            layers.append(("B", rB, zB, probB))

        for tag, r, zz, prob in layers:
            print(f"\n  [{tag}층]  BSS {bss(yv, prob(zz)):.1f}")
            print(f"  {'형질':<22}{'dev t':>8}{'LEAG t':>8}{'PERM t':>8}"
                  f"{'lvl t':>8}{'판정':>6}")
            for x in TRAITS:
                _, t = resid_t(r, feat["dev_" + x])
                _, tl = resid_t(r, feat["devLEAG_" + x])
                _, tp = resid_t(r, feat["devPERM_" + x])
                _, tv2 = resid_t(r, feat["lvl_" + x])
                keep.setdefault((tag, x), []).append((t, tl, tp))
                mk = "✓" if (abs(t) > 2 and abs(t) > 2 * max(abs(tl), abs(tp))) else ""
                print(f"  dev_{x:<18}{t:>8.2f}{tl:>8.2f}{tp:>8.2f}{tv2:>8.2f}{mk:>6}")

            # β 스윕은 참고 (dev만)
            base = bss(yv, prob(zz))
            print(f"\n  [{tag}층] β 스윕 dev_* (ΔBSS vs {base:.1f})")
            print("  " + f"{'형질':<22}" + "".join(f"{b:>8.3f}" for b in BETAS))
            for x in TRAITS:
                v = feat["dev_" + x]
                xc = (v - v.mean()) / max(float(v.std()), 1e-9)
                row = [bss(yv, prob(zz + b * xc)) - base for b in BETAS]
                print("  " + f"dev_{x:<18}" + "".join(f"{q:>8.2f}" for q in row))

    print("\n" + "=" * 78)
    print("판정 — 양 fold(A층) 부호 일치 + |t|>2 + 위약(max) 2배")
    print("=" * 78)
    print(f"  {'형질':<22}{'A t23':>8}{'A t24':>8}{'B t24':>8}{'부호':>6}{'위약2배':>8}")
    for x in TRAITS:
        (t1, l1, p1), (t2, l2, p2) = keep[("A", x)]
        b = keep.get(("B", x))
        same = (t1 > 0) == (t2 > 0)
        over = (abs(t1) > 2 and abs(t1) > 2 * max(abs(l1), abs(p1))
                and abs(t2) > 2 and abs(t2) > 2 * max(abs(l2), abs(p2)))
        btxt = f"{b[0][0]:>8.2f}" if b else "      —"
        print(f"  dev_{x:<18}{t1:>8.2f}{t2:>8.2f}{btxt}"
              f"{'✓' if same else '✗':>6}{'✓' if over else '✗':>8}")


if __name__ == "__main__":
    main()
