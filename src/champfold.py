"""§7.213 — **검증된 챔피언 파이프라인** fold 층에서 이 세션 수치를 전부 다시 낸다.

§7.212b 가 레시피를 hj03 파일에 직접 대조해 확정했다 (corr 1.00000000 · max|Δz| 1.3e-15).
§7.212 가 그 레시피로 fold 코어 20멤버 × 2 fold 를 학습했다 — s29 계보 12 대비
**f2023 +9.1 · f2024 +10.4** (corr .986/.996). 즉 §7.198~§7.211 은 **다른 모델** 위였다.

조립 (hj03 `predict.py` 실물 순서)
    z  = mean(코어 LGBM 20)                                   ← cc_lgb_* (챔피언 레시피)
    z  = (1−α)·z + α·(0.875·emb4 + 0.125·noid)                α = 0.75
    z  = (1−0.10)·z + 0.10·mean(CatBoost 12)
    z += logit_adjust(ctx: hand 1.875 · count4 1.25,
                      asof: gap_success .03 · gap_bsuccess .03 · **gap_ball −0.0038**)
    p  = shift_to_prior(σ(z), ȳ)

⚠️ 재현 못 하는 것 (채연 계보, 학습 코드 없음): compact paired delta · general LGBM 12 ·
   user_fspec 라우팅. 예측 가중 약 11%. 세 팔에 **동일하게** 빠지므로 팔 간 비교는 유효하다.

다시 내는 것
    ① emb_dim 4→8, 콜드 패치 (D4 / D8 / D8C)
    ② α 스윕
    ③ gap 축 β 스윕 + 엔티티 순열 위약 + 투수 클러스터 부트스트랩
    ④ `runner_on_1b` ctx δ 축

실행: .venv/Scripts/python.exe -u src/champfold.py 2>&1 | tee experiments/champfold.log
"""
import sys
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
import ctxfit
import deploy29 as D
import features as F
import hillclimb as H
from config import (CTX_K, MLP_ALPHA, MLP_SEEDS, RECENCY_HALF_LIFE, TARGET_COL,
                    TRAIN_CSV)
from data import load_csv, recency_weight
from gapstrike import EXTRA_SPEC
from mixresid import GAP_SPEC, build
from validation import bss, shift_to_prior

CACHE = D.CACHE
VARIANTS = ("base", "wide", "extra", "lowff")
CORE_SEEDS = (42, 7, 2024, 1234, 31337)
ARMS = ("D4", "D8", "D8C")
# ⚠️ **config.py 를 보지 않는다.** 챔피언이 정하는 값은 hj03 artifacts 가 진실원이다.
#    config.CTX_AXES 는 hand β 1.875 인데 챔피언은 **1.054** 다 (혜진 T1 값).
#    그걸 config 에서 가져다 쓰는 바람에 첫 조립이 −18.8 을 맞았다.
HJ = Path("D:/t1eval/HJ03")
_A = joblib.load(HJ / "model" / "artifacts.joblib")["artifacts"]
CTX_DEPLOY = [(a["name"], float(a["beta"])) for a in _A["ctx"]]
ASOF_DEPLOY = [(a["name"], float(a["beta"])) for a in _A["asof"]]
ASOF_S = float(_A["asof"][0]["strength"])
ALPHAS = [0.65, 0.70, 0.75, 0.80, 0.85]
GAPS = ["gap_strike", "gap_middle", "gap_bmiddle", "gap_reverse"]
GBETAS = [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
CBETAS = [0.25, 0.5, 1.0, 1.5, 2.25]
B_BOOT = 500


def _code_set1b(df):
    v = pd.to_numeric(df["runner_on_1b"], errors="coerce").fillna(0)
    return (v.to_numpy() > 0).astype(np.int64), 2


def fold_state(df, s, y_all, fold):
    st = D.state(df, s, y_all, fold, check=False)
    mm, yv, vi = st["mm"], st["yv"], st["vi"]
    core = np.mean([np.load(CACHE / f"cc_lgb_f{fold}_{v}_{sd}.npz")["z"]
                    for v in VARIANTS for sd in CORE_SEEDS], axis=0)[mm]
    cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                   for v in H.CAT_VARIANTS for sd in H.CAT_SEEDS], axis=0)[mm]
    noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]

    # δ — **`jointbeta` 경로(`D.state`)를 쓴다.** 배포 실측 앵커(807.2/924.5)를 재현하는
    #    검증된 경로다. 내가 hj03 artifacts 명세(hand 1.054 등)로 손수 조립한
    #    `ctxfit`+`asofdelta` 판은 corr 0.77/0.85 로 **다른 물건**이었고 −14~−25 였다
    #    (§7.213 진단). 혜진의 δ 표 구성은 저장소 코드로 재현되지 않는다.
    tr = df[s < fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    d = df.iloc[vi[mm]].copy().reset_index(drop=True)
    ax = np.asarray(st["Xax"] @ st["dep"], np.float64)
    return st, mm, yv, d, core, cat, noid, ax, tr, w


def main():
    for k, v in EXTRA_SPEC.items():
        asofdelta.ASOF_SPEC.setdefault(k, v)
    F.CTX_CODERS.setdefault("set1b", _code_set1b)
    print("δ = jointbeta 배포 경로 (앵커 807.2/924.5 재현). "
          f"참고 — hj03 artifacts 명세: ctx {CTX_DEPLOY} · asof {ASOF_DEPLOY}")
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)

    for fold in (2023, 2024):
        st, mm, yv, d, core, cat, noid, ax, tr, w = fold_state(df, s, y_all, fold)
        r = float(yv.mean())

        def emb(arm):
            e = [np.load(CACHE / f"champ_{arm}_f{fold}_emb4_{sd}.npz")["z"]
                 for sd in MLP_SEEDS]
            return (0.875 * np.mean(e, axis=0) + 0.125 * np.mean(noid, axis=0))[mm]

        def blend(zm, a=MLP_ALPHA):
            return (1 - H.W_CAT) * ((1 - a) * core + a * zm) + H.W_CAT * cat + ax

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        print("=" * 96)
        print(f"fold {fold}   행 {len(yv):,}   ⚠️ train < {fold} — 홀드아웃 · "
              f"코어=챔피언 레시피 20 · δ 5축")
        print("=" * 96)

        # ① emb_dim · 콜드 패치
        Z, S = {}, {}
        for a in ARMS:
            Z[a] = blend(emb(a))
            S[a] = sc(Z[a])
        print(f"  ① D4 {S['D4']:.2f}   D8 {S['D8']:.2f} (Δ {S['D8']-S['D4']:+.2f})"
              f"   D8C {S['D8C']:.2f} (Δ vs D8 {S['D8C']-S['D8']:+.2f})")

        # ② α 스윕
        # ⚠️ fold 점수로 팔을 고르지 않는다 — **챔피언 구성**이 기준이다:
        #    emb_dim 8 + 콜드 패치 = D8C. (fold argmax 로 고르면 선택 편향이고
        #    양 fold 가 다른 팔이 되어 크기비를 못 읽는다.)
        best = "D8C"
        zm = emb(best)
        print(f"  ② α 스윕 (팔 {best}):  " + "  ".join(
            f"{a:.2f}→{sc(blend(zm, a)):.2f}" for a in ALPHAS))

        z0 = Z[best]
        base = S[best]
        # 축 재료
        art = asofdelta.fit_offsets(tr)
        league = float(tr[TARGET_COL].mean())
        priors = {c: float(np.nanmean(df.loc[s < fold, c].to_numpy(np.float64)))
                  for _, c, _ in GAP_SPEC}
        feat = build(d, art, league, priors)
        rng = np.random.default_rng(0)
        dsh = d.copy()
        dsh["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
        dsh["batter_id"] = rng.permutation(d["batter_id"].to_numpy())
        fsh = build(dsh, art, league, priors)

        codes, uniq = pd.factorize(d["pitcher_id"].to_numpy())
        order = np.argsort(codes, kind="stable")
        bnd = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        br = np.random.default_rng(7)
        IDX = [np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                               br.choice(seen, size=len(seen), replace=True)])
               for _ in range(B_BOOT)]

        def judge(x, xs, betas, tag):
            row = [sc(z0 + b * x) - base for b in betas]
            i = int(np.argmax(row))
            b, dv = betas[i], row[i]
            pl = sc(z0 + b * xs) - base
            zc = z0 + b * x
            bs = np.array([bss(yv[j], shift_to_prior(1 / (1 + np.exp(-zc[j])), r))
                           - bss(yv[j], shift_to_prior(1 / (1 + np.exp(-z0[j])), r))
                           for j in IDX])
            lo = float(np.percentile(bs, 2.5))
            print(f"     {tag:<14}" + "".join(f"{v:>+8.2f}" for v in row)
                  + f"{b:>+8.3f}{dv:>+8.2f}{pl:>+8.2f}{lo:>+9.2f}")

        print(f"  ③ gap 축 (기준 {best} {base:.2f})")
        print(f"     {'축':<14}" + "".join(f"{b:>+8.3f}" for b in GBETAS)
              + f"{'β*':>8}{'Δ':>8}{'위약':>8}{'95%하한':>9}")
        for c in GAPS:
            if c in feat:
                judge(feat[c].astype(np.float64), fsh[c].astype(np.float64),
                      GBETAS, c)

        # ④ runner_on_1b ctx δ
        axes1b = ctxfit.fit_ctx(tr, TARGET_COL, [("set1b", 1.0)], CTX_K, weights=w)
        x1 = F.ctx_lookup(d, axes1b[0]).astype(np.float64)
        xs1 = F.ctx_lookup(dsh, axes1b[0]).astype(np.float64)
        print(f"  ④ {'축':<14}" + "".join(f"{b:>+8.2f}" for b in CBETAS)
              + f"{'β*':>8}{'Δ':>8}{'위약':>8}{'95%하한':>9}")
        judge(x1, xs1, CBETAS, "runner_on_1b")
        print()


if __name__ == "__main__":
    main()
