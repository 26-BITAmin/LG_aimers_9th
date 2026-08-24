"""§7.139 후속 — **죽은 축 대조군이 왜 안 죽었나.** init_score x cat_features 의 2x2.

§7.139 에서 사전 등록한 눈금이 fold2024 에서 무너졌다:
    allid (pitcher_id 711 + batter_id 742 를 CTR 로) = §7.12 가 -101.4 를 낸 그 조작인데
    fold2024 핸디캡이 +6.9 (기준선 -10.6 대비 **+17.5**) 로 **안 죽었다.**
대조군이 후보와 같이 오르면 fold2024 의 양수(+6.4 / +18.8 / +13.7)를 해석할 수 없다.

가설 (검정 가능):
    §7.12 의 기전은 명시돼 있다 — "pitcher_id CTR 은 실력이 아니라 **어느 시대에
    던졌나**를 인코딩한다" (시즌 base rate 가 .565 -> .486 로 흐르므로).
    그런데 §7.111 이 나중에 `init_score`(시즌x게임타입 레벨)를 도입했고(LB +5.30),
    §7.139 는 공정성을 위해 CatBoost 에도 `baseline=init_tr` 을 줬다.
    **init_score 가 시즌 레벨을 흡수하면 CTR 이 시대를 인코딩할 이유가 사라진다.**
    -> §7.12 의 진단은 그때 맞았고, §7.111 이 원인을 이미 제거했을 수 있다.

2x2 (fold 2개 x 2 팔 x init on/off = 8칸)
    가설이 참이면:  init OFF 에서 allid 가 none 대비 **크게 음수** (-101 방향 재현)
                    init ON  에서 그 격차가 사라지거나 뒤집힘  <- §7.139 관측
    가설이 거짓이면: init on/off 가 allid-none 격차를 안 바꾼다
                    -> 그러면 §7.139 fold2024 의 양수는 다른 원인이고, 눈금은 여전히 없다.

다른 것은 전부 §7.139 와 동일하다 (같은 X, soften, GPU, 3시드, 250 iter).
init OFF 는 baseline 만 뺀다 — soften 은 그대로 둔다 (오염 보정은 별개 축이다).

실행:  .venv/Scripts/python.exe -u src/catinit.py 2>&1 | tee experiments/catinit.log
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
from catboost import CatBoostClassifier, Pool

import deploy29 as D
from catgpu import CAT_PARAMS, ITERS, LOWCARD, SEEDS, prep
from config import MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
ALLID = LOWCARD + ["pitcher_id", "batter_id"]
CELLS = [("none", []), ("allid", ALLID)]


def train(Xtr, Xva, y2, w2, init_tr, cats, tag, fold, use_init, t0):
    key = f"catinit_fold{fold}_{tag}_init{int(use_init)}.npz"
    pth = CACHE / key
    if pth.exists():
        return np.load(pth)["z"]
    Xtr, Xva = Xtr.copy(), Xva.copy()
    idx = None
    if cats:
        for c in cats:
            Xtr[c] = Xtr[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
            Xva[c] = Xva[c].to_numpy(np.float64).round().astype(np.int64).astype(str)
        idx = [Xtr.columns.get_loc(c) for c in cats]
    z, n = np.zeros(len(Xva)), 0
    for sd in SEEDS:
        m = CatBoostClassifier(iterations=ITERS, random_seed=sd, **CAT_PARAMS)
        m.fit(Pool(Xtr, y2, weight=w2,
                   baseline=(init_tr if use_init else None), cat_features=idx))
        zz = m.predict(Pool(Xva, cat_features=idx), prediction_type="RawFormulaVal")
        z += zz
        n += 1
        del m
    z /= n
    np.savez_compressed(pth, z=z)
    print(f"    fold {fold} [{tag} init={int(use_init)}] 완료 ({time.time() - t0:.0f}s)")
    return z


def main():
    t0 = time.time()
    df, s, y = D.load_df()
    res = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep = st["mm"], st["yv"], st["dep"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]
        z_blend = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp

        def sc(zb, solo=False):
            z = zb[mm] + (0.0 if solo else Xa[mm] @ dep)
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), yv.mean()))

        base = sc(z_blend)
        assert abs(base - D.EXPECT_BASE[fold]) < 0.3
        solo_lgb = sc(z_lgb, solo=True)
        print(f"\nfold {fold}  앵커 OK {base:.1f}  ·  LGBM solo {solo_lgb:.1f}")
        Xtr, Xva, y2, w2, init_tr = prep(df, fold)
        for use_init in (True, False):
            for tag, cats in CELLS:
                zc = train(Xtr, Xva, y2, w2, init_tr, cats, tag, fold, use_init, t0)
                sol = sc(zc, solo=True)
                d30 = sc(0.7 * z_blend + 0.3 * zc) - base
                res[(fold, tag, use_init)] = (sol, sol - solo_lgb, d30)
                print(f"    {tag:<6s} init={int(use_init)}  solo {sol:8.1f}"
                      f"  핸디캡 {sol - solo_lgb:+8.1f}  블렌드Δ(w=.30) {d30:+7.1f}")

    print("\n" + "=" * 96)
    print("판정 — allid − none 격차가 init_score 유무로 바뀌는가")
    print("=" * 96)
    print(f"  {'fold':>6s}{'init':>6s}{'none solo':>12s}{'allid solo':>12s}"
          f"{'allid−none':>13s}{'none Δ.30':>11s}{'allid Δ.30':>12s}")
    for fold in D.FOLDS:
        for ui in (False, True):
            n_s, _, n_d = res[(fold, "none", ui)]
            a_s, _, a_d = res[(fold, "allid", ui)]
            print(f"  {fold:>6d}{int(ui):>6d}{n_s:>12.1f}{a_s:>12.1f}"
                  f"{a_s - n_s:>+13.1f}{n_d:>+11.1f}{a_d:>+12.1f}")
    print("\n  가설 참: init=0 에서 allid−none 이 크게 음수(§7.12 −101 방향),")
    print("           init=1 에서 그 격차가 사라진다 → §7.111 이 원인을 제거한 것.")
    print("  가설 거짓: init 이 격차를 안 바꾼다 → §7.139 fold2024 양수는 눈금 없는 상태.")


if __name__ == "__main__":
    main()
