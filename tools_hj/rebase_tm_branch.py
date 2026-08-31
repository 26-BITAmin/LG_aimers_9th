"""제출 zip 에 **TrackMan CatBoost 잔차 브랜치**를 추가한다 — §7.240 → 빌드.

    corr = mean_seed CatBoost(공식 numeric 37 + TM 프로필 29 + cat 8)
    p   += γ · corr · 1{game_type=='R' ∧ ctx 프로필 존재}     (아핀 이후, clip 이전)
    PROB_SHIFT 재보정으로 Δmean ≈ 0

학습: 챔피언 등가 fold 층 잔차 (f2023 + f2024 **풀링** — 단일 fold 적합 금지 메모리).
    각 fold 의 TM 피처는 as-of(<fold), 서빙 프로필은 ≤2024 (as-of 2025) — 동형 규율.
서빙: (pitcher_id, count_index, batter_right) 정적 조회표 + 행 자신 컬럼만 — 행 독립.

■ 측정 근거 (§7.240 v2): 정직 칸 B−base +4.05@γ0.10 (G3 +4.14 [−1.54,+9.78] P .920),
   TM 증분 +2.46 [+0.31,+4.57] P .983 (0 배제). sanity +4.55@γ0.15. γ 보수측 0.10.

■ 구조 검증 (이 도구가 직접 assert)
   ① 비적용행(F ∨ 미커버) Δp == 0  ② 적용행 Δp 와 오프라인 corr 재현 일치
   ③ 변경 격리 = predict.py 수정 + model/ 신규 4파일  ④ 재보정 후 Δmean ≈ 0

사용
  python tools_hj/rebase_tm_branch.py --src hj16_psucc.zip --out hj19_tmcb.zip --gamma 0.10
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd

from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha

import asofdelta
from config import SEED, TARGET_COL
from data import load_csv
from gapstrike import EXTRA_SPEC
from tmaxis import (GROUPS, PHYS, TM_PARQ, build_tm_features, champ_col_axes,
                    ctx_keys, share_table, wmean_table)
from tmbranch import (CAT_COLS, CB_PARAMS, SEEDS_CB, fold_ctx,
                      official_numeric_cols)

GAMMA_DEFAULT = 0.10

INIT_OLD = """        self.ctx_hits = {}
"""
INIT_NEW = """        self.ctx_hits = {}
        # TrackMan CatBoost 잔차 브랜치 (§7.240) — 상태가 없으면 꺼진다.
        _tm = os.path.join(model_dir, "tmcb_state.joblib")
        self.tmcb = joblib.load(_tm) if os.path.exists(_tm) else None
        self.tmcb_models = []
"""

APPLY_OLD = """        p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER) + PROB_SHIFT
        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"""
APPLY_NEW = """        p = PROB_CENTER + PROB_SCALE * (p - PROB_CENTER) + PROB_SHIFT
        # ── TrackMan CatBoost 잔차 브랜치 (§7.240) — 아핀 이후, clip 이전 ──
        # (pitcher_id, count_index, batter_right) 정적 조회표(≤2024 train 파생)와
        # 행 자신의 컬럼만 쓴다 — test 행 간 정보 없음 (CLAUDE.md §3.2).
        # game_type 은 라우팅 게이트로만 쓴다 (§7.101 선례).
        if self.tmcb is not None:
            import pandas as pd
            st = self.tmcb
            if not self.tmcb_models:
                from catboost import CatBoostRegressor
                for fn in st["model_files"]:
                    m = CatBoostRegressor()
                    m.load_model(os.path.join(self.model_dir, fn))
                    self.tmcb_models.append(m)
            balls = pd.to_numeric(test_df["balls_before"],
                                  errors="coerce").fillna(0).clip(0, 3)
            strikes = pd.to_numeric(test_df["strikes_before"],
                                    errors="coerce").fillna(0).clip(0, 2)
            hand = test_df["batter_hand"].astype(str).str.strip().str.lower()
            key = pd.DataFrame({
                "pid": pd.to_numeric(test_df["pitcher_id"], errors="coerce")
                       .astype("Int64").astype(str).to_numpy(),
                "cidx": (balls.astype("int8") * 3
                         + strikes.astype("int8")).to_numpy(),
                "bright": hand.isin(["2", "r", "right", "우"])
                          .astype("int8").to_numpy()})
            mc = key.merge(st["ctx"], on=["pid", "cidx", "bright"],
                           how="left", sort=False)
            mp = key.merge(st["pit"], on="pid", how="left", sort=False)
            covered = np.isfinite(mc["w"].to_numpy(dtype=float))
            is_r = test_df["game_type"].to_numpy().astype(str) == "R"
            mask = covered & is_r
            self.tmcb_rate = float(mask.mean())
            if mask.any():
                T = pd.DataFrame(index=key.index)
                for c in st["phys"]:
                    T["ctx_" + c] = mc[c].to_numpy(dtype=float)
                    T["pit_" + c] = mp[c].to_numpy(dtype=float)
                    T["gap_" + c] = T["ctx_" + c] - T["pit_" + c]
                for g in st["groups"]:
                    T["ctx_share_" + g] = mc["share_" + g].to_numpy(dtype=float)
                T["ctx_logw"] = np.log1p(mc["w"].to_numpy(dtype=float))
                T["pit_logw"] = np.log1p(mp["w"].to_numpy(dtype=float))
                Xb = pd.DataFrame(index=key.index)
                for c in st["num_cols"]:
                    Xb[c] = (pd.to_numeric(test_df[c], errors="coerce")
                             if c in test_df.columns else np.nan)
                for c in T.columns:
                    Xb[c] = T[c]
                Xb = Xb.replace([np.inf, -np.inf], np.nan)
                for c in Xb.columns:
                    Xb[c] = Xb[c].fillna(st["fill"][c])
                for c in st["cat_cols"]:
                    Xb[c] = (pd.Series(test_df[c].to_numpy())
                             .astype("string").fillna("MISSING")
                             .astype(str).to_numpy())
                Xb = Xb[st["feature_cols"]]
                sub = Xb.iloc[np.flatnonzero(mask)]
                cs = np.zeros(len(sub))
                for m in self.tmcb_models:
                    cs += m.predict(sub)
                corr = np.zeros(len(test_df))
                corr[mask] = cs / len(self.tmcb_models)
                p = p + float(st["gamma"]) * corr
        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"""


def read_text(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def write_text(p, s):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def patch(p, pairs):
    s = read_text(p)
    eol = "\r\n" if "\r\n" in s else "\n"
    for old, new in pairs:
        old, new = old.replace("\n", eol), new.replace("\n", eol)
        n = s.count(old)
        assert n == 1, f"{os.path.basename(p)}: 패턴 {n}개 — {old[:60]!r}"
        s = s.replace(old, new)
    write_text(p, s)


def pid_norm(v):
    return pd.to_numeric(pd.Series(np.asarray(v)), errors="coerce").astype(
        "Int64").astype(str).to_numpy()


def build_serving_state(jt, num_cols, fill, feature_cols, gamma, model_files):
    """≤2024 전체로 as-of 2025 프로필 표를 동결한다."""
    hist = jt[jt["season"] < 2025]
    w = np.power(0.80, 2024 - hist["season"].to_numpy(np.float64))
    kk = ["pid", "cidx", "bright"]
    ctx = wmean_table(hist, kk, w)
    ctx = ctx.merge(share_table(hist, kk, w), on=kk, how="left")
    pit = wmean_table(hist, ["pid"], w)
    ctx["pid"] = pid_norm(ctx["pid"])
    pit["pid"] = pid_norm(pit["pid"])
    return dict(ctx=ctx, pit=pit, phys=PHYS, groups=GROUPS,
                num_cols=num_cols, cat_cols=CAT_COLS,
                feature_cols=feature_cols, fill=fill, gamma=float(gamma),
                model_files=model_files)


def offline_corr(st, frame):
    """패치와 같은 로직의 오프라인 재현 (검증 대조용)."""
    from catboost import CatBoostRegressor
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).clip(0, 3)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).clip(0, 2)
    hand = frame["batter_hand"].astype(str).str.strip().str.lower()
    key = pd.DataFrame({
        "pid": pid_norm(frame["pitcher_id"]),
        "cidx": (balls.astype("int8") * 3 + strikes.astype("int8")).to_numpy(),
        "bright": hand.isin(["2", "r", "right", "우"]).astype("int8").to_numpy()})
    mc = key.merge(st["ctx"], on=["pid", "cidx", "bright"], how="left", sort=False)
    mp = key.merge(st["pit"], on="pid", how="left", sort=False)
    covered = np.isfinite(mc["w"].to_numpy(float))
    is_r = frame["game_type"].to_numpy().astype(str) == "R"
    mask = covered & is_r
    T = pd.DataFrame(index=key.index)
    for c in st["phys"]:
        T["ctx_" + c] = mc[c].to_numpy(float)
        T["pit_" + c] = mp[c].to_numpy(float)
        T["gap_" + c] = T["ctx_" + c] - T["pit_" + c]
    for g in st["groups"]:
        T["ctx_share_" + g] = mc["share_" + g].to_numpy(float)
    T["ctx_logw"] = np.log1p(mc["w"].to_numpy(float))
    T["pit_logw"] = np.log1p(mp["w"].to_numpy(float))
    Xb = pd.DataFrame(index=key.index)
    for c in st["num_cols"]:
        Xb[c] = pd.to_numeric(frame[c], errors="coerce") if c in frame.columns else np.nan
    for c in T.columns:
        Xb[c] = T[c]
    Xb = Xb.replace([np.inf, -np.inf], np.nan)
    for c in Xb.columns:
        Xb[c] = Xb[c].fillna(st["fill"][c])
    for c in st["cat_cols"]:
        Xb[c] = (pd.Series(frame[c].to_numpy()).astype("string")
                 .fillna("MISSING").astype(str).to_numpy())
    Xb = Xb[st["feature_cols"]]
    from catboost import CatBoostRegressor as _C
    corr = np.zeros(len(frame))
    if mask.any():
        sub = Xb.iloc[np.flatnonzero(mask)]
        cs = np.zeros(len(sub))
        for pth in st["_model_paths"]:
            m = _C()
            m.load_model(pth)
            cs += m.predict(sub)
        corr[mask] = cs / len(st["_model_paths"])
    return corr, mask


def main():
    from catboost import CatBoostRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    a_ = ap.parse_args()

    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    deployed = champ_col_axes(ROOT / a_.src)
    print(f"§7.240 빌드 — TM CatBoost 잔차 브랜치  γ {a_.gamma:.2f}")

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
    jt = pd.concat([jt.drop(columns=base_cols[2:]), ctx_keys(jt)], axis=1)
    num_cols = official_numeric_cols(df)

    # ── 풀링 학습 행렬 (f2023 + f2024, 각자 as-of TM) ────────────
    frames, resids = [], []
    for fold in (2023, 2024):
        c = fold_ctx(df, s, y_all, fold, jt, deployed, rng)
        m = c["is_r"] & c["covered"]
        d_m = c["d"][m].reset_index(drop=True)
        T_m = c["T"][m].reset_index(drop=True)
        Xf = pd.DataFrame(index=range(len(d_m)))
        for col in num_cols:
            Xf[col] = pd.to_numeric(d_m[col].to_numpy(), errors="coerce")
        for col in T_m.columns:
            Xf[col] = T_m[col].to_numpy(np.float64)
        Xf["_catsrc"] = np.arange(len(d_m))
        frames.append((Xf, d_m))
        resids.append((c["yv"] - c["p0"])[m])
    X_num = pd.concat([f[0].drop(columns=["_catsrc"]) for f in frames],
                      ignore_index=True)
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    fill = {c: float(md) if np.isfinite(md := X_num[c].median()) else 0.0
            for c in X_num.columns}
    for c in X_num.columns:
        X_num[c] = X_num[c].fillna(fill[c])
    for c in CAT_COLS:
        vals = np.concatenate([f[1][c].to_numpy() for f in frames])
        X_num[c] = (pd.Series(vals).astype("string").fillna("MISSING")
                    .astype(str).to_numpy())
    y_res = np.concatenate(resids)
    feature_cols = list(X_num.columns)
    cat_idx = [X_num.columns.get_loc(c) for c in CAT_COLS]
    print(f"풀링 학습 {len(X_num):,}행 × {len(feature_cols)}피처  "
          f"resid sd {y_res.std():.5f}")

    model_files = []
    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="tmcb_", dir="D:/t1eval")
    mcache = ROOT / "experiments" / "tmcb_models"
    mcache.mkdir(exist_ok=True)
    for sd in SEEDS_CB:
        fn = f"tmcb_seed{sd}.cbm"
        cached = mcache / fn
        if cached.exists():
            shutil.copy(cached, os.path.join(tmp, fn))
            print(f"  seed {sd}  캐시 재사용")
        else:
            t0 = time.time()
            mdl = CatBoostRegressor(random_seed=sd, **CB_PARAMS)
            mdl.fit(X_num, y_res, cat_features=cat_idx)
            mdl.save_model(os.path.join(tmp, fn))
            shutil.copy(os.path.join(tmp, fn), cached)
            print(f"  seed {sd}  {(time.time()-t0)/60:.1f}분")
        model_files.append(fn)

    st = build_serving_state(jt, num_cols, fill, feature_cols,
                             a_.gamma, model_files)
    st["_model_paths"] = [os.path.join(tmp, fn) for fn in model_files]
    print(f"서빙 프로필: ctx {len(st['ctx']):,}셀 · pit {len(st['pit']):,}투수")

    # ── zip 패치 ────────────────────────────────────────────
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a_.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)
    pp = os.path.join(work, "predict.py")
    shift0 = float(SHIFT_RE.findall(read_text(pp))[0][1])
    patch(pp, [(INIT_OLD, INIT_NEW), (APPLY_OLD, APPLY_NEW)])
    print(f"패치  predict.py (init + affine 이후 브랜치)  PROB_SHIFT {shift0:.9f}")

    proxy = df[df["season"] == 2024].drop(
        columns=[TARGET_COL]).reset_index(drop=True)
    pq = os.path.join(tmp, "proxy.pkl")
    proxy.to_pickle(pq)

    print("추론 ① 브랜치 상태 없음 (패치만) …")
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    st_save = {k: v for k, v in st.items() if not k.startswith("_")}
    joblib.dump(st_save, os.path.join(work, "model", "tmcb_state.joblib"))
    for fn in model_files:
        shutil.copy(os.path.join(tmp, fn), os.path.join(work, "model", fn))

    print("추론 ② 브랜치 ON (레벨 미보정) …")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
    dd = mid - before
    corr_off, mask_off = offline_corr(st, proxy)
    dm = float(dd.mean())
    mx_off = float(np.abs(dd[~mask_off]).max()) if (~mask_off).any() else 0.0
    print(f"  적용행 {mask_off.mean():.3f} · Δp 평균 {dm:+.6g} · "
          f"max|Δp| {np.abs(dd).max():.6g} · 비적용행 max|Δp| {mx_off:.3g}")
    assert np.abs(dd).max() > 0.0, "브랜치가 예측에 반영되지 않았다"
    assert mx_off == 0.0, "게이트 실패 — 비적용행이 움직였다"
    tgt = a_.gamma * corr_off[mask_off]
    got = dd[mask_off]
    cr = float(np.corrcoef(got, tgt)[0, 1])
    mad = float(np.abs(got - tgt).max())
    print(f"  적용행 corr(Δp, γ·corr_off) {cr:+.6f} · max|차이| {mad:.3g}")
    assert cr > 0.999 and mad < 1e-6, "오프라인 재현 불일치"

    shift = shift0 - dm
    write_text(pp, SHIFT_RE.sub(lambda m: f"{m.group(1)}{shift:.9f}",
                                read_text(pp)))
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")

    print("추론 ③ 재보정 검증 …")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    rr = float((after - before).mean())
    print(f"  재보정 후 Δmean {rr:+.3g}  {'OK' if abs(rr) < 1e-7 else '경고'}")
    assert abs(rr) < 1e-7

    all_files = sorted(
        str(Path(r) / f).replace("\\", "/").split("work/", 1)[1]
        for r, _, fs in os.walk(work) for f in fs
        if "__pycache__" not in r and not f.endswith(".pyc"))
    with zipfile.ZipFile(a_.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in all_files:
            z.write(os.path.join(work, n), n)
    changed = sorted(n for n in names if os.path.isfile(os.path.join(orig, n))
                     and open(os.path.join(work, n), "rb").read()
                     != open(os.path.join(orig, n), "rb").read())
    added = sorted(set(all_files) - set(names))
    print(f"\n변경 {changed} · 추가 {added}")
    assert changed == ["predict.py"], f"기대 밖 변경: {changed}"
    assert added == sorted(["model/tmcb_state.joblib"]
                           + [f"model/{f}" for f in model_files])
    assert len(os.path.basename(a_.out)) <= 30
    print(f"출력   {a_.out}\n  SHA-256 {sha(a_.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
