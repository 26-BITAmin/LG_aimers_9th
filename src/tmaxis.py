"""§7.239 **TrackMan 프로필 δ 축** — other_team 행 단위 매칭 테이블 기반 (`src/tmaxis.py`).

base: `hj16_psucc.zip` 등가 fold 층 (§7.232~§7.238 과 동일 하네스, 아핀 ON).

■ 재료 — other_team/Aimers `trackman_joined.parquet` (1,285,301행, row_id → 물리량 8종)
    검증 완료: 표본 30,000행의 물리량 8-튜플이 대회 제공 `trackman_history.csv`와
    100% 일치, train 시즌 정합 99.77%, row_id 중복 0. 커버리지 87.1% (R행 88.2%).
    즉 제공 데이터에서 파생 가능한 물건 — 외부 데이터 아님.

■ 왜 다시 여는가 — 트랙맨은 §7.9/§7.39/§7.116a 로 세 번 닫힌 채널이다.
    재개봉 근거 = **새 도구** (행 단위 매칭; 우리 §4 매핑은 투수 단위였다)
    + **레짐 신호** (그 팀이 2025 LB 실측으로 백본 투입 +20 을 확인 — 1097→1117.
    단 그들의 베이스는 1097, 우리는 1148 이라 전이 기대는 훨씬 작다).
    메모리 준수: "레짐 단절 신호 없이 세 번째 재측정 금지" — 신호가 생겼으므로 1회 재측정.

■ 리키지 가드
    검증 행 자신의 매칭 물리량은 절대 사용하지 않는다. 프로필은 season < fold 행으로만
    만들고 decay 0.80 (그 팀의 채택값) recency 가중. 2025 추론과 동형의 as-of 규율.

■ 사전 등록 (PRE 1):
    tm_ridge   물리 프로필 29피처 → 잔차 ridge 합성축. **cross-fold 적합**:
               fold2024 축은 fold2023 잔차로 적합 (시간 순방향 = 정직),
               fold2023 축은 fold2024 잔차로 적합 (시간 역방향 = sanity 참고).
■ 탐색적 (EXPLORE 4):
    tm_velo    투수 구속 프로필 (rel_speed 가중평균)
    tm_gap     컨텍스트 프로필의 투수 전체 프로필 대비 표준화 이탈 크기 (제구 적응 프록시)
    tm_fb      컨텍스트 fastball 조성 몫
    tm_w       컨텍스트 노출량 log1p (커버리지 축 — 대조 성격)
■ 위약: P:tm_ridge (투수 정체 순열 — count·타자 구조는 보존) · 널: 죽은 컬럼 8 선형
■ 관문 — §7.237 과 동일 (G0 앵커 assert · G1 부호 · G2 널바닥 · G3 부트 · G4 고원 · G5 전이)
    + f2023 함정 감시: 이득이 f2023 에만 몰리면 §7.9 프로파일 재현으로 기각.
■ 재베이스 — (pitcher_id, count_index, batter_right) 조회표 + 선형 계수.
    `rebase_asof_axis.py` 기하 (조회형). 행 독립 — test groupby 없음.

실행: .venv/Scripts/python.exe -u src/tmaxis.py 2>&1 | tee experiments/tmaxis.log
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd

import asofdelta
import champfold as CF
import champspec as CS
import contam_fit
import features as F
import hillclimb as H
import modelabel
from colaxis import ANCHOR, ANCHOR_BETA, ANCHOR_EXPECT, std_axis
from config import (CONTAM_AXES, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, ROOT,
                    SEED, SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from gapstrike import EXTRA_SPEC
from grevaxis import PLATEAU_FRAC, plateau_beta
from validation import bss, shift_to_prior

CHAMP = ROOT / "hj16_psucc.zip"
TM_PARQ = Path(r"C:\Users\강원우\AppData\Local\Temp\claude"
               r"\D--projects-LG-aimers---"
               r"\90b6928a-72cc-4e49-92fa-97c7e5481d95\scratchpad"
               r"\other_team\Aimers\data\trackman_joined.parquet")
FOLDS = (2023, 2024)
N_TRAIN_SUB = 400_000
B_BOOT = 1000
TM_DECAY = 0.80
RIDGE_ALPHA = 1000.0

PRE = ["tm_ridge"]
EXPLORE = ["tm_velo", "tm_gap", "tm_fb", "tm_w"]
NULL_COLS = ["outs_before", "strikes_before", "num_runners_on", "is_first_pitch",
             "is_late_close", "top_bottom", "game_dayofweek", "game_month"]

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
GROUPS = ["fastball", "breaking", "offspeed"]

FINE = np.array([-0.0140, -0.0120, -0.0100, -0.0085, -0.0070, -0.0060, -0.0050,
                 -0.0040, -0.0030, -0.0025, -0.0020, -0.0015, -0.0010, -0.0005,
                 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050,
                 0.0060, 0.0070, 0.0085, 0.0100, 0.0120, 0.0140])
WIDE = np.array(sorted(set(np.round(FINE, 4)) |
                       {-0.12, -0.08, -0.05, -0.03, -0.02,
                        0.02, 0.03, 0.05, 0.08, 0.12}))


def to_prob(praw, r):
    return CS.prob_transform(praw, CS.solve_shift(praw, r))


def champ_col_axes(zip_path):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extract("model/artifacts.joblib", td)
        art = joblib.load(Path(td) / "model" / "artifacts.joblib")["artifacts"]
    return [(a["name"], float(a["beta"])) for a in (art.get("col") or [])]


def ctx_keys(frame):
    """(pitcher_id, count_index, batter_right) — jt/d 양쪽에 같은 함수로 적용해
    인코딩 방향과 무관하게 키 일관성을 보장한다."""
    balls = pd.to_numeric(frame["balls_before"], errors="coerce")
    balls = balls.fillna(0).clip(0, 3).astype(np.int8)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce")
    strikes = strikes.fillna(0).clip(0, 2).astype(np.int8)
    hand = frame["batter_hand"].astype(str).str.strip().str.lower()
    right = hand.isin(["2", "r", "right", "우"]).astype(np.int8)
    return pd.DataFrame({
        "pid": frame["pitcher_id"].astype(str).to_numpy(),
        "cidx": (balls * 3 + strikes).to_numpy(),
        "bright": right.to_numpy(),
    }, index=frame.index)


def wmean_table(hist, keys, w):
    t = hist[keys].copy()
    t["_w"] = w
    for c in PHYS:
        v = hist[c].to_numpy(np.float64)
        t[f"n_{c}"] = np.where(np.isfinite(v), v, 0.0) * w
        t[f"d_{c}"] = np.isfinite(v) * w
    g = t.groupby(keys, sort=False).sum()
    out = pd.DataFrame(index=g.index)
    out["w"] = g["_w"]
    for c in PHYS:
        den = g[f"d_{c}"].to_numpy()
        out[c] = np.where(den > 0, g[f"n_{c}"].to_numpy() / np.where(den > 0, den, 1.0),
                          np.nan)
    return out.reset_index()


def share_table(hist, keys, w):
    m = hist["pitch_type_group"].isin(GROUPS)
    t = hist.loc[m, keys + ["pitch_type_group"]].copy()
    t["_w"] = w[m.to_numpy()]
    piv = t.pivot_table(index=keys, columns="pitch_type_group", values="_w",
                        aggfunc="sum", fill_value=0.0)
    for gname in GROUPS:
        if gname not in piv.columns:
            piv[gname] = 0.0
    tot = piv[GROUPS].sum(axis=1).to_numpy()
    out = pd.DataFrame(index=piv.index)
    for gname in GROUPS:
        out[f"share_{gname}"] = np.where(tot > 0, piv[gname].to_numpy() / np.where(
            tot > 0, tot, 1.0), np.nan)
    return out.reset_index()


def build_tm_features(jt, d, fold):
    """season<fold 매칭 행으로 프로필을 만들고 valid 프레임 d 에 조회한다.
    반환: (T DataFrame [n_valid × 29], covered bool)"""
    hist = jt[jt["season"] < fold]
    w = np.power(TM_DECAY, (fold - 1) - hist["season"].to_numpy(np.float64))

    kk = ["pid", "cidx", "bright"]
    ctx = wmean_table(hist, kk, w)
    ctx = ctx.merge(share_table(hist, kk, w), on=kk, how="left")
    pit = wmean_table(hist, ["pid"], w)

    dk = ctx_keys(d)
    mc = dk.merge(ctx, on=kk, how="left", sort=False)
    mp = dk.merge(pit, on="pid", how="left", sort=False,
                  suffixes=("", "_p"))
    assert len(mc) == len(d) and len(mp) == len(d)

    T = pd.DataFrame(index=d.index)
    for c in PHYS:
        T[f"ctx_{c}"] = mc[c].to_numpy(np.float64)
        T[f"pit_{c}"] = mp[c].to_numpy(np.float64)
        T[f"gap_{c}"] = T[f"ctx_{c}"] - T[f"pit_{c}"]
    for gname in GROUPS:
        T[f"ctx_share_{gname}"] = mc[f"share_{gname}"].to_numpy(np.float64)
    T["ctx_logw"] = np.log1p(mc["w"].to_numpy(np.float64))
    T["pit_logw"] = np.log1p(mp["w"].to_numpy(np.float64))
    covered = np.isfinite(mc["w"].to_numpy(np.float64))
    return T, covered


def standardize_block(T, covered):
    """covered 행 기준 표준화, 비커버·NaN → 0 (중립)."""
    X = T.to_numpy(np.float64).copy()
    mu = np.nanmean(X[covered], axis=0)
    sd = np.nanstd(X[covered], axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    X = (X - mu) / sd
    X[~np.isfinite(X)] = 0.0
    X[~covered] = 0.0
    return X


def make_sc(yv, r):
    def sc(z):
        return bss(yv, to_prob(1 / (1 + np.exp(-z)), r))
    return sc


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    deployed = champ_col_axes(CHAMP)
    print(f"§7.239  TrackMan 프로필 δ 축 — {CHAMP.name} 등가 층, 아핀 ON")
    print(f"  재료: other_team trackman_joined.parquet (row_id 매칭, decay {TM_DECAY})")
    print(f"  PRE 1 (tm_ridge, cross-fold 적합) + EXPLORE 4 · 위약 1 · "
          f"널 {len(NULL_COLS)} · B_BOOT {B_BOOT}\n")

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED)

    jt = pd.read_parquet(TM_PARQ)
    jt["row_id"] = jt["row_id"].astype(str)
    base_cols = ["row_id", "season", "pitcher_id", "batter_hand",
                 "balls_before", "strikes_before"]
    key_src = df[base_cols].copy()
    key_src["row_id"] = key_src["row_id"].astype(str)
    jt = jt.merge(key_src, on="row_id", how="inner")
    jk = ctx_keys(jt)
    jt = pd.concat([jt.drop(columns=base_cols[2:]), jk], axis=1)
    print(f"  parquet 조인 후 {len(jt):,}행 (train 대비 {len(jt)/len(df):.1%})\n")

    curves, vs, ctxs, tmstate = {}, {}, {}, {}

    for fold in FOLDS:
        st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
        r = float(yv.mean())
        E = [np.load(CF.CACHE / f"champ_D8C_f{fold}_emb4_{sd}.npz")["z"]
             for sd in MLP_SEEDS]
        zm = (0.875 * np.mean(E, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]
        z0 = ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
              + H.W_CAT * cat + ax)

        sc = make_sc(yv, r)

        lab, _, _ = modelabel.recover(tr)
        rev = np.where(np.isnan(lab["reverse"]), 0.0, np.round(lab["reverse"]))
        ct = contam_fit.fit_contam(tr, rev)
        a_r = asofdelta.fit_asof_axes(tr, [(ANCHOR, 1.0)], CF.ASOF_S, ct=ct)[0]
        v_r = F.asof_lookup(d, a_r, ct=ct).astype(np.float64)
        pr0 = 1 / (1 + np.exp(-z0))
        prA = 1 / (1 + np.exp(-(z0 + ANCHOR_BETA * v_r)))
        d_ns = bss(yv, shift_to_prior(prA, r)) - bss(yv, shift_to_prior(pr0, r))
        err = abs(d_ns - ANCHOR_EXPECT[fold]) / abs(ANCHOR_EXPECT[fold])
        print("=" * 118)
        print(f"fold {fold}  G0 앵커 D {d_ns:+.2f} (오차 {err:.2%}) "
              f"{'OK' if err < 0.02 else 'FAIL'}")
        assert err < 0.02, "G0 실패"
        z0 = z0 + ANCHOR_BETA * v_r

        wt = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                            ref_season=int(tr["season"].max()))
        art = F.fit_artifacts(tr, weights=wt, k_pitcher=SMOOTH_K_PITCHER,
                              k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
        Xv = F.build_features(d, art)
        idx = np.sort(rng.choice(len(tr), size=min(N_TRAIN_SUB, len(tr)),
                                 replace=False))
        Xt = F.build_features(tr.iloc[idx], art)

        def axis_of(col):
            t = np.asarray(Xt[col], dtype=np.float64)
            t = t[np.isfinite(t)]
            return std_axis(np.asarray(Xv[col], dtype=np.float64),
                            float(t.mean()), float(t.std()))

        for nm, b in deployed:
            z0 = z0 + b * axis_of(nm)
        base = sc(z0)
        print(f"  기준선 {base:.2f} (= 챔피언 등가)")

        # ── TM 프로필 피처 ─────────────────────────────────────
        T, covered = build_tm_features(jt, d, fold)
        X = standardize_block(T, covered)
        print(f"  TM 커버리지 (valid): ctx {covered.mean():.1%} · "
              f"pit {np.isfinite(T['pit_logw'].to_numpy()).mean():.1%}")

        p0 = to_prob(1 / (1 + np.exp(-z0)), r)
        resid = yv - p0
        n_cov = int(covered.sum())
        tt = (X[covered] * resid[covered, None]).sum(0) / np.sqrt(
            (X[covered] ** 2).sum(0).clip(1e-9)) / resid[covered].std()
        top = np.argsort(-np.abs(tt))[:8]
        print("  잔차 프로브 t (상위 8): "
              + "  ".join(f"{T.columns[i]}:{tt[i]:+.1f}" for i in top))

        nulls = {f"N:{c}": axis_of(c) for c in NULL_COLS}
        ctxs[fold] = dict(z0=z0, base=base, yv=yv, r=r, d=d, sc=sc)
        tmstate[fold] = dict(T=T, X=X, covered=covered, resid=resid,
                             nulls=nulls, n_cov=n_cov)
        print("=" * 118)

    # ── cross-fold ridge 합성 + 축 구성 ───────────────────────
    other = {2023: 2024, 2024: 2023}
    coefs = {}
    for src in FOLDS:
        stt = tmstate[src]
        Xc = stt["X"][stt["covered"]]
        rc = stt["resid"][stt["covered"]]
        A_ = Xc.T @ Xc + RIDGE_ALPHA * np.eye(Xc.shape[1])
        coefs[src] = np.linalg.solve(A_, Xc.T @ rc)
    cols = list(tmstate[FOLDS[0]]["T"].columns)
    for src in FOLDS:
        top = np.argsort(-np.abs(coefs[src]))[:6]
        print(f"  ridge(적합 f{src}) 상위 계수: "
              + "  ".join(f"{cols[i]}:{coefs[src][i]:+.5f}" for i in top))

    for fold in FOLDS:
        stt = tmstate[fold]
        cx = ctxs[fold]
        z0, base = cx["z0"], cx["base"]
        sc = make_sc(cx["yv"], cx["r"])
        X, T, covered = stt["X"], stt["T"], stt["covered"]
        prng = np.random.default_rng(SEED + fold)

        def stdv(v):
            t = v[np.isfinite(v)]
            return std_axis(v, float(t.mean()), float(t.std()))

        comp = X @ coefs[other[fold]]
        gap_cols = [i for i, c in enumerate(cols) if c.startswith("gap_")]
        A = {
            "tm_ridge": stdv(comp),
            "tm_velo": stdv(np.where(covered, T["pit_rel_speed"].to_numpy(), np.nan)),
            "tm_gap": stdv(np.where(covered,
                                    np.abs(X[:, gap_cols]).mean(axis=1), np.nan)),
            "tm_fb": stdv(np.where(covered,
                                   T["ctx_share_fastball"].to_numpy(), np.nan)),
            "tm_w": stdv(np.where(covered, T["ctx_logw"].to_numpy(), np.nan)),
        }
        # 위약 — 투수 정체 순열: d 의 프로필 값 자체를 투수 블록 단위로 섞는 대신,
        # 합성축 값을 투수 단위로 재배치 (count·타자 구조 공유 항 보존)
        pid = ctxs[fold]["d"]["pitcher_id"].astype(str).to_numpy()
        uniq = pd.unique(pid)
        perm = dict(zip(uniq, prng.permutation(uniq)))
        pid_perm = np.array([perm[p] for p in pid])
        pm = pd.DataFrame({"pid": pid_perm}).merge(
            pd.DataFrame({"pid": pid, "v": comp}).groupby("pid", sort=False)
            .mean().reset_index(), on="pid", how="left")
        A["P:tm_ridge"] = stdv(pm["v"].to_numpy(np.float64))
        A.update(stt["nulls"])

        cands = PRE + EXPLORE
        tag = ("[정직: 적합 f2023→평가 f2024]" if fold == 2024
               else "[sanity: 적합 f2024→평가 f2023, 시간 역방향]")
        print(f"\nfold {fold}  tm_ridge {tag}")
        print(f"  {'축':<16}{'종류':>6}{'beta*':>10}{'D@beta*':>10}{'G4':>4}"
              f"   0 근처 곡선")
        for name in list(A):
            v = A[name]
            dd = np.array([sc(z0 + b * v) - base for b in WIDE])
            bs = plateau_beta(WIDE, dd, PLATEAU_FRAC)[0]
            g4 = bs is not None
            j = int(np.argmin(np.abs(WIDE - bs))) if g4 else int(np.argmax(dd))
            curves.setdefault(name, {})[fold] = (WIDE.copy(), dd, float(WIDE[j]),
                                                 float(dd[j]))
            if name in cands:
                vs.setdefault(fold, {})[name] = v
            kind = ("사전등록" if name in PRE
                    else ("탐색적" if name in EXPLORE
                          else ("위약" if name.startswith("P:") else "널")))
            ii = [int(np.argmin(np.abs(WIDE - b)))
                  for b in (-0.01, -0.0025, -0.0005, 0.0005, 0.0025, 0.01)]
            head = "  ".join(f"{WIDE[i]:+.4f}:{dd[i]:+.2f}" for i in ii)
            print(f"  {name:<16}{kind:>6}{WIDE[j]:>+10.4f}{dd[j]:>+10.2f}"
                  f"{'Y' if g4 else 'n':>4}   {head}")

    nb = {f: max(curves[c][f][3] for c in curves if c.startswith("N:"))
          for f in FOLDS}
    pb = {f: {c: round(curves[c][f][3], 2) for c in curves if c.startswith("P:")}
          for f in FOLDS}
    print(f"\n  널바닥(선형): f2023 {nb[2023]:+.2f} / f2024 {nb[2024]:+.2f}")
    print(f"  위약:        f2023 {pb[2023]}   f2024 {pb[2024]}")

    cx = ctxs[2024]
    z0, yv, r, d = cx["z0"], cx["yv"], cx["r"], cx["d"]
    codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    seen = np.where(np.diff(bnd) > 0)[0]
    br_ = np.random.default_rng(SEED)
    idxs = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                            br_.choice(seen, size=len(seen), replace=True)])
            for _ in range(B_BOOT)]
    p0 = to_prob(1 / (1 + np.exp(-z0)), r)
    print(f"\n  G3 투수 클러스터 부트스트랩 (fold2024, B={B_BOOT})")
    for c in PRE + EXPLORE:
        if c not in vs.get(2024, {}):
            continue
        b = curves[c][2024][2]
        pa = to_prob(1 / (1 + np.exp(-(z0 + b * vs[2024][c]))), r)
        dif = np.array([bss(yv[i], pa[i]) - bss(yv[i], p0[i]) for i in idxs])
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"    {c:<16}beta*={b:+.4f}  D {curves[c][2024][3]:+7.2f}  "
              f"95% [{lo:+7.2f}, {hi:+7.2f}]  P(D>0) {(dif > 0).mean():.3f}")

    print("\n  G5 시즌 전이 (f2023 beta* -> f2024 채점, 그 반대)")
    print(f"    {'축':<16}{'f23 b*':>9}{'f23 D':>8}{'f24 b*':>9}{'f24 D':>8}"
          f"{'23->24':>9}{'24->23':>9}  판정")
    for name in PRE + EXPLORE + sorted(c for c in curves
                                       if c.startswith(("N:", "P:"))):
        if name not in curves or len(curves[name]) < 2:
            continue
        b23, d23, s23, v23 = curves[name][2023]
        b24, d24, s24, v24 = curves[name][2024]
        f_ = float(d24[int(np.argmin(np.abs(b24 - s23)))])
        w_ = float(d23[int(np.argmin(np.abs(b23 - s24)))])
        ok = ((np.sign(s23) == np.sign(s24)) and v24 > nb[2024]
              and v23 > nb[2023] and f_ > 0 and w_ > 0)
        kind = ("사전등록" if name in PRE
                else ("탐색적" if name in EXPLORE
                      else ("위약" if name.startswith("P:") else "널")))
        print(f"    {name:<16}{s23:>+9.4f}{v23:>+8.2f}{s24:>+9.4f}{v24:>+8.2f}"
              f"{f_:>+9.2f}{w_:>+9.2f}  {'PASS' if ok else 'fail'}  [{kind}]")


if __name__ == "__main__":
    main()
