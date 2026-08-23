"""**타자 키 δ** — 저장소에 없던 채널 (`MODEL.md §7.129`).

`ctxfit.fit_ctx` 는 `pitcher_id` 로만 키를 만든다 (`ctxfit.py:49`).
채택된 두 축(`hand` +5.93, `count4` +3.08)도 전부 **투수 × 상황**이다.
**타자 × 상황은 시도된 적이 없다.**

저장소 자신의 판정 기준이 이 자리를 가리킨다 —
> 플래툰 δ가 채택된 이유: `asof_*` 에 손잡이 분할이 **없어서** 잔차에 신호가 남아 있었다.

`asof_batter_*` 는 컬럼이 **3개**(n · success_rate · middle_rate)뿐인데 투수 쪽은
**15개**다. 타자 쪽이 훨씬 더 뭉개져 있다. 그리고 `gap_bsuccess`(타자 시즌 진행분)가
이미 채택돼 있으므로 **타자 신호 자체는 실재한다.**

세 팔을 나란히 잰다 (memory: 새 피처는 잔차에 먼저 물어라 — 주효과 대조군 필수).
    axis      δ(batter, code)              후보
    main      δ(batter, ·) 상황 무시        **주효과 대조군** — 타자 레벨은 이미 모델에 있나
    placebo   δ(batter, 코드 셔플)          같은 자유도, 상황 정보 0
두 fold 가 같은 방향이어야 후보다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import deploy29 as D                                            # noqa: E402
from config import CTX_K, TARGET_COL                            # noqa: E402
from features import CTX_CODERS                                 # noqa: E402


def _code_phand(df):
    """**투수** 손잡이. 타자 키와 짝지어야 타자 플래툰이 된다.

    ⚠️ 배포 축 `hand` 는 `_code_hand` = **타자** 손잡이다. 그것은 `pitcher_id` 와
       짝지어 투수 플래툰(§7.36, LB +5.93)이 된다. 같은 코더를 `batter_id` 와
       짝지으면 **키가 거의 1:1** 이라 δ 가 항등적으로 0 에 가깝다 (실측 std 0.0023).
    """
    return df["pitcher_hand"].to_numpy(dtype=np.int64), 8
from validation import bss, shift_to_prior                      # noqa: E402


def fit_delta(tr, y, w, keycol, code, m, k):
    """δ(entity, code) = 수축 × (셀 성공률 − 그 엔티티 전체 성공률).  fit_ctx 와 같은 식."""
    eid = tr[keycol].to_numpy(dtype=np.int64)
    ue, inve = np.unique(eid, return_inverse=True)
    sw_e = np.bincount(inve, weights=w)
    rate_e = np.bincount(inve, weights=w * y) / np.maximum(sw_e, 1e-12)
    key = eid * m + code
    u, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sw2 = np.bincount(inv, weights=w * w)
    n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
    t = np.bincount(inv, weights=w * y)
    re = rate_e[np.searchsorted(ue, u // m)]
    return u, (n_eff / (n_eff + k)) * (t / np.maximum(sw, 1e-12) - re)


def lookup(d, keycol, code, m, u, delta):
    key = d[keycol].to_numpy(dtype=np.int64) * m + code
    j = np.clip(np.searchsorted(u, key), 0, max(len(u) - 1, 0))
    return np.where((len(u) > 0) & (u[j] == key), delta[j], 0.0)


def main(seed=0):
    df, s, y = D.load_df()
    rng = np.random.default_rng(seed)
    print(f"CTX_K = {CTX_K}   (배포 δ 와 같은 수축)")
    res = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        d, z0, yv = st["d"], st["z0"], st["yv"]
        r = float(yv.mean())
        tr = df[s < fold]
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        w = np.ones(len(ytr))

        def sc(dz):
            p = 1 / (1 + np.exp(-(z0 + dz)))
            return bss(yv, shift_to_prior(np.clip(p, 1e-6, 1 - 1e-6), r)) - st["base"]

        print(f"\nfold {fold}  기준 BSS {st['base']:.2f}")
        print(f"  {'축':<22s}{'셀':>9s}{'δ std':>9s}" +
              "".join(f"{'β='+str(b):>9s}" for b in (0.5, 1.0, 1.875, 3.0)))
        for name in ("phand", "count4"):
            coder = _code_phand if name == "phand" else CTX_CODERS[name]
            ctr, m = coder(tr); cva, _ = coder(d)
            arms = {
                f"batter×{name}": (ctr, cva),
                f"  main(상황무시)": (np.zeros_like(ctr), np.zeros_like(cva)),
                f"  placebo(셔플)": (rng.permutation(ctr), cva),
            }
            for tag, (c1, c2) in arms.items():
                u, dl = fit_delta(tr, ytr, w, "batter_id", c1, m, CTX_K)
                dv = lookup(d, "batter_id", c2, m, u, dl)
                row = "".join(f"{sc(b*dv):>+9.2f}" for b in (0.5, 1.0, 1.875, 3.0))
                res.setdefault(tag.strip(), {})[fold] = sc(1.875 * dv)
                print(f"  {tag:<22s}{len(u):>9,}{dl.std():>9.5f}{row}")
    print()
    print("두 fold 대조 (β=1.875):")
    for k, v in res.items():
        ok = "✅" if min(v.values()) > 0 else "❌"
        print(f"  {k:<22s} f2023 {v.get(2023,float('nan')):>+8.2f}   "
              f"f2024 {v.get(2024,float('nan')):>+8.2f}   {ok}")


if __name__ == "__main__":
    main()
