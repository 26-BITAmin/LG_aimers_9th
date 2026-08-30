"""콜드스타트 패치의 **홀드아웃** 효과 — 투수 · 타자 · 둘 다 (§7.209)

왜 이게 필요한가
    §7.208 은 `sy06_bcold.zip` 을 빌드하고 "기전과 노출만 확정, 효과는 예보 안 함" 으로
    끝냈다. **그건 게으른 결론이다.** 이 저장소에는 배포 등가 홀드아웃 층이 있다
    (§7.194 B1, 앵커 807.68 / 925.75). 거기서 재면 된다.

★★ 홀드아웃이 마스킹 프록시보다 **낫다**
    §7.203·§7.208 은 2024 프록시에 미지 개체가 0명이라 인위적으로 `id = −1` 마스킹을 했다.
    fold 는 그럴 필요가 없다 — fold f 의 검증 시즌에는 **진짜 신규 개체**가 들어 있다
    (시즌 < f 로 학습했으므로). 즉 **실제 콜드스타트 상황을 그대로 잰다.**
    마스킹이 흔드는 `asof_*` 교락도 없다.

무엇을 하나
    ① fold 별로 emb4 5멤버를 **가중치까지 저장하며** 재학습 (foldcache 는 z 만 저장한다)
    ② 하네스 검증 — 재현한 z 가 `foldcache` 캐시 멤버와 일치하는가
    ③ 4개 팔: base / pitch(emb0[0]) / bat(emb1[0]) / both  → 각각 순전파
    ④ B1 블렌드(LGBM 12 캐시 + MLP 8 + CatBoost 12 캐시) → 배포 δ → oracle 레벨 → BSS
    ⑤ 투수 클러스터 부트스트랩

⚠️ 잡음 눈금 (§7.199): MLP 층 재학습 SD(Δ) 4.8 / 7.1.
   이 패치는 **재학습이 아니라 이미 학습된 가중치의 1행 수정**이므로 그 잡음이 안 낀다
   (base 와 patched 가 **같은 학습 결과**를 공유한다 — 완전 쌍체).
   → 여기서 나오는 Δ 는 draw 잡음이 아니라 패치 자체의 효과다.

실행:  .venv/Scripts/python.exe -u src/bcoldfold.py 2>&1 | tee experiments/bcoldfold.log
"""
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
import modelabel
import soften
from config import (CONTAM_AXES, DROP_FEATURES, MLP_ALPHA, MLP_AUX_LAMBDA, MLP_SEEDS,
                    MLP_VARIANTS, RECENCY_HALF_LIFE, ROOT, SMOOTH_K_BATTER,
                    SMOOTH_K_PITCHER, TARGET_COL, USE_PLAYER_IDS)
from data import recency_weight
from features import build_features, fit_artifacts
from tmeval import W_CAT, boot, load_family
from validation import bss, shift_to_prior

CACHE = ROOT / "experiments" / "cache"
FOLDS = [2023, 2024]
EMB4_SEEDS = MLP_SEEDS[:MLP_VARIANTS["emb4"][1]]        # 5개

# ★★ 챔피언은 `pitcher_id` 임베딩이 **dim 8** 이다 (§7.197 — 그것이 LB +7.44 의
#    유일한 실질 변경이었다). 그런데 `config.py` 의 MLP_VARIANTS 는 아직 **4** 다
#    (§7.202 가 "meta.json 이 emb_dim 4 로 거짓말" 이라고 적은 것과 같은 틈).
#    → B1 fold 하네스가 재현하는 807.68/925.75 는 **dim 4 앵커**다.
#    콜드 패치는 임베딩 표를 직접 건드리므로 이 차이가 측정에 직결된다.
#    두 값을 다 돌려 보고한다. [[audit-inherited-constants-not-just-registered-gates]]
P_DIM = int(os.environ.get("LGA_P_DIM", "8"))          # 8 = 챔피언 / 4 = config·B1 앵커
ARMS = ["base", "pitch", "bat", "both"]
RNG_SEED = 42


def prep(df, fold):
    """foldcache.compute 의 MLP 경로와 같다 — aux 라벨 · soften(aux) · init_score 없음."""
    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES,
                        center_asof=False)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    aux, _cov = modelabel.aux_labels(tr)
    Xtr, ytr, w2, aux, _h = soften.soften(tr, Xtr, y_tr, w, aux)
    return art, Xtr, ytr, w2, aux, Xva


def train_members(df, fold, t0):
    """emb4 5멤버를 **가중치까지** 저장하며 학습. 반환: (npz dict 목록, enc, Xva)."""
    import mlp
    art, Xtr, ytr, w2, aux, Xva = prep(df, fold)
    enc = mlp.fit_encoder(Xtr, art)
    A, Ctr = mlp.encode_apply(Xtr, enc)
    B, Cva = mlp.encode_apply(Xva, enc)
    sizes = enc["sizes"]
    cfg = dict(MLP_VARIANTS["emb4"][0])
    cfg["emb_dim"] = dict(cfg["emb_dim"], pitcher_id=P_DIM)

    out = []
    for sd in EMB4_SEEDS:
        p = CACHE / f"bcoldw_d{P_DIM}_f{fold}_emb4_{sd}.npz"
        if p.exists():
            out.append(dict(np.load(p, allow_pickle=True)))
            print(f"    가중치 재사용 f{fold}/{sd}  ({time.time()-t0:.0f}s)")
            continue
        m, curve = mlp.train_mlp(A, Ctr, ytr, w2, B, Cva, sizes, seed=sd,
                                 verbose=False, y_aux=aux,
                                 **dict(cfg, aux=MLP_AUX_LAMBDA))
        wts = mlp.export_weights(m)
        wts["_z"] = curve[-1]                 # 학습 경로가 낸 z (대조용)
        np.savez_compressed(p, **wts)
        out.append(wts)
        del m
        gc.collect()
        print(f"    학습 f{fold}/{sd}  ({time.time()-t0:.0f}s)")
    return out, enc, Xva


def patched_z(wts, enc, Xva, arm):
    """arm 에 맞게 임베딩 0행을 무게중심으로 바꾸고 순전파."""
    from mlp_infer import MLPPredictor
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in wts.items()
         if not k.startswith("_")}
    ecols = [str(c) for c in d["ecols"]]
    for col, tag in (("pitcher_id", "pitch"), ("batter_id", "bat")):
        if arm in (tag, "both") and col in ecols:
            k = f"emb{ecols.index(col)}"
            e = d[k]
            d[k] = np.vstack([e[1:].mean(0)[None, :], e[1:]]).astype(e.dtype)
    return MLPPredictor(d, enc).logit(Xva)


def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    print(f"§7.209 콜드스타트 패치 홀드아웃 — 지문 {D.KEY}   pitcher_id emb_dim = {P_DIM}"
          f"  ({'챔피언 구조' if P_DIM == 8 else 'config·B1 앵커 구조'})")
    print("★ 마스킹 프록시가 아니다. fold 검증 시즌의 **진짜 신규 개체**로 잰다.\n")
    df, s, y = D.load_df()

    for fold in FOLDS:
        st = D.state(df, s, y, fold)
        st["ybar"] = float(st["yv"].mean())
        mm = st["mm"]

        # 이 fold 의 실제 콜드스타트 노출 — 프록시가 아니라 실측
        tr, va = df[s < fold], df[s == fold]
        pnew = ~va["pitcher_id"].isin(set(tr["pitcher_id"].unique())).to_numpy()
        bnew = ~va["batter_id"].isin(set(tr["batter_id"].unique())).to_numpy()
        print(f"{'='*88}")
        print(f"■ fold {fold}   검증 {len(va):,}행 (clean {mm.sum():,})   "
              f"신규 투수 행 {pnew.mean():.2%} · 신규 타자 행 {bnew.mean():.2%}")
        print(f"{'='*88}")

        wts, enc, Xva = train_members(df, fold, t0)

        # ── 하네스 검증: 재현 z 가 foldcache 멤버와 맞는가 ──────────
        cached = load_family("base", "mlp", fold)          # (n, 8) — emb4 5 + noid 3
        rec = np.stack([patched_z(w, enc, Xva, "base") for w in wts], axis=1)
        mad = float(np.abs(rec - cached[:, :5]).max())
        if P_DIM == 4:
            # foldcache 는 config 값(dim 4)으로 학습돼 있다 → 비트 대조가 가능하다
            print(f"  재현 검증 max|Δz| = {mad:.3e}  "
                  f"{'OK' if mad < 1e-4 else '❌ 하네스 어긋남'}")
            assert mad < 1e-4, "재현 실패 — 이 아래 숫자를 믿을 수 없다"
        else:
            # dim 8 은 참조 캐시가 없다. 대신 **다른 모델임이 확실한지**를 확인한다
            # (같으면 emb_dim 덮어쓰기가 안 먹은 것이다 — 조용히 dim 4 를 재게 된다).
            assert mad > 1e-3, (
                f"dim {P_DIM} 인데 foldcache(dim4) 와 z 가 같다 (max|Δz| {mad:.2e}) "
                "— emb_dim 덮어쓰기가 안 먹었다")
            print(f"  dim {P_DIM} — foldcache(dim4) 대비 max|Δz| = {mad:.3e} (달라야 정상)")
            print(f"  emb0 shape = {wts[0]['emb0'].shape}  (챔피언 (793, 8) 과 같은 차원)")

        z_lgb = load_family("base", "lgb", fold).mean(axis=1)
        z_cat = load_family("base", "cat", fold).mean(axis=1)
        z_noid = cached[:, 5:].mean(axis=1)
        ax = st["Xax"] @ st["dep"]

        def score(zemb4):
            zm = 0.875 * zemb4.mean(axis=1) + 0.125 * z_noid
            z = (1 - W_CAT) * ((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm) + W_CAT * z_cat
            p = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), st["ybar"])
            return bss(st["yv"], p), p

        Z = {}
        for arm in ARMS:
            Z[arm] = (rec if arm == "base"
                      else np.stack([patched_z(w, enc, Xva, arm) for w in wts], axis=1))
        base_s, base_p = score(Z["base"])
        print(f"  B1(base) = {base_s:.2f}   (deploy29 앵커 {st['base']:.1f})")
        print(f"  {'팔':<8}{'BSS':>10}{'Δ':>9}{'95% CI':>20}{'P(Δ>0)':>9}"
              f"{'미지행 Δp̄':>13}{'알려진행 Δp̄':>14}")
        for arm in ARMS[1:]:
            sc, p = score(Z[arm])
            lo, hi, pg = boot(st, p, base_p, rng)
            dp = p - base_p
            tgt = (pnew if arm == "pitch" else bnew if arm == "bat" else (pnew | bnew))[mm]
            print(f"  {arm:<8}{sc:>10.2f}{sc-base_s:>+9.2f}   [{lo:+7.2f},{hi:+7.2f}]"
                  f"{pg:>9.3f}{dp[tgt].mean():>+13.2e}{dp[~tgt].mean():>+14.2e}")
        del Xva, wts, Z
        gc.collect()
        print()

    print("=" * 88)
    print("읽는 법")
    print("=" * 88)
    print("  · base 와 patched 는 **같은 학습 결과**를 공유한다 (1행 수정) → 완전 쌍체.")
    print("    §7.199 의 MLP 재학습 잡음 SD 4.8/7.1 이 여기엔 **안 낀다**.")
    print("  · 알려진 행 Δp̄ 가 0 이 아니면 패치가 엉뚱한 행까지 건드린 것이다")
    print("    (임베딩 0행은 미지 행만 타므로 정확히 0 이어야 한다).")
    print("  · 양 fold 부호가 갈리면 채택 근거가 아니다 — §7.9 가 그 자리에서 LB −13.6 을 냈다.")


if __name__ == "__main__":
    main()
