"""§7.149 — **R-only 라우팅**: CatBoost 를 F 행에서 빼면 이득이 커지는가.

근거 (§7.145-④): s35 등가 층 fold2024 에서 가법 분해가
    dTotal +2.467 = dF **−0.534** + dR +3.000,  초과분 −0.826 [−1.647, −0.029]
로 **F 집중이 확정**됐다. dF 를 되돌리면 약 +0.5 다.

기전 가설: **fspec 이 이미 F 행 전문가**라, 거기에 범용 멤버를 10% 섞으면 전문성이
희석된다. 이 저장소는 이미 "F 행은 전문가에게 라우팅한다"를 채택했고(§7.101),
범용 멤버를 F 에 태우는 것이 오히려 예외적이다.

⚠️ **그러나 이것은 fold 결과를 보고 나온 규칙이다.** 서브그룹 튜닝이고, 이 저장소는
   무의미한 버킷(요일)으로 쪼갰을 때 +22.8 이 나온 전례가 있다. 그래서 **위약이
   하나가 아니라 분포**여야 한다.

#### 설계

    후보    F 행(game_type=="F") 을 CatBoost 블렌드에서 제외
    위약    **죽은 축**의 버킷을 같은 방식으로 제외 — 요일 7개 + 투수해시 5개 = 12개
            (요일·월은 §7.11 에서 드랍된 확인된 죽은 축이다)
    기준선  현행 배포 = 전 행에 α=0.10

#### ★★ fold2023 이 내장 대조군이다

기전은 **fspec 이 있어야** 성립한다. fspec 은 fold2024 에만 정의된다
(학습셋 `F & season>=2023`, fold2023 train 은 ≤2022 라 깨끗한 F 0행).
    기전이 참이면   fold2024 에서만 효과, fold2023 은 위약 대역
    기전이 거짓이면 두 fold 에서 비슷 -> "CatBoost 가 F 에서 그냥 나쁘다"
**두 시나리오가 정반대 그림을 그린다.** 이게 위약보다 강한 판정자다.

#### 채택 기준 — 실행 전에 박는다

  (1) fold2024(s35 층) R-only Δ > 0 이고 부트스트랩 구간이 0 배제 (B=8000)
  (2) **R-only Δ 가 위약 12개 분포의 최대값을 넘을 것** (단순 상위가 아니라 최대)
  (3) fold2023(s29 층, fspec 없음) 에서는 위약 대역 안에 있을 것 — 기전 정합
      ⚠️ (3) 은 "커야 좋다"가 아니다. **작아야** 기전이 맞다.
  (4) 크기비는 요구하지 않는다 — 기전상 두 fold 가 다를 것을 예측하기 때문이다

실행:  .venv/Scripts/python.exe -u src/catronly.py 2>&1 | tee experiments/catronly.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import deploy29 as D
from catens import SEEDS, VARIANTS
from config import FSPEC_ALPHA, MLP_ALPHA
from validation import bss, shift_to_prior

CACHE = Path(__file__).parent.parent / "experiments" / "cache"
W, B = 0.10, 8000


def arms(dd, isf):
    """(이름, 제외 마스크, 노출율). 위약은 죽은 축(요일)과 투수해시."""
    out = [("F (후보)", isf)]
    dow = dd["game_dayofweek"].to_numpy(np.int64)
    for k in sorted(np.unique(dow)):
        out.append((f"위약 요일={k}", dow == k))
    pid = dd["pitcher_id"].to_numpy(np.int64)
    # 투수 단위 해시 버킷 — F 노출(약 11.8%)에 맞춰 17분할 중 2개를 묶는다
    hb = (pid * 2654435761 % 17)
    for j in range(5):
        out.append((f"위약 투수해시={j*2}", (hb == j * 2)))
    return [(n, m, float(m.mean())) for n, m in out]


def main():
    df, s, y = D.load_df()
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        with np.load(CACHE / f"fold{fold}_{D.KEY}.npz") as d0:
            z_mlp = d0["z_mlp"]
        z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
        mm, yv, dep, cl, n_id = st["mm"], st["yv"], st["dep"], st["cl"], st["n_id"]
        Xa = np.zeros((len(z_lgb), st["Xax"].shape[1]))
        Xa[mm] = st["Xax"]
        ax = Xa[mm] @ dep
        ybar = yv.mean()
        ref = ybar * (1 - ybar)
        dd = st["d"]
        isf = dd["game_type"].to_numpy().astype(str) == "F"
        ens = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                       for v in VARIANTS for sd in SEEDS], axis=0)

        z29 = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
        fs = CACHE / f"s35_fspec_fold{fold}.npz"
        has_fspec = fs.exists()
        if has_fspec:
            zb = z29.copy()
            vf = (df[s == fold]["game_type"].to_numpy().astype(str) == "F")
            zb[vf] = (1 - FSPEC_ALPHA) * z29[vf] + FSPEC_ALPHA * np.load(fs)["z"][vf]
            layer = "s35 등가 층 (fspec 있음)"
        else:
            zb, layer = z29, "s29 층 (**fspec 없음** — 기전상 효과가 없어야 정상)"

        # 기준선 = 현행 배포: 전 행에 α=0.10
        zc_all = (1 - W) * zb + W * ens
        base_p = shift_to_prior(1 / (1 + np.exp(-(zc_all[mm] + ax))), ybar)
        base = bss(yv, base_p)
        e0 = (base_p - yv) ** 2
        n = np.bincount(cl, minlength=n_id)
        s0 = np.bincount(cl, weights=e0, minlength=n_id)
        seen = np.where(n > 0)[0]
        print(f"\nfold {fold} · {layer}")
        print(f"  기준선(전 행 α={W}) = {base:.2f}   F 노출 {isf.mean():.1%}")
        print(f"  {'팔':<18s}{'노출':>7s}{'Δ':>8s}{'2.5%':>8s}{'97.5%':>8s}{'P(Δ>0)':>8s}")
        res = {}
        for name, mask, expo in arms(dd, isf):
            z = zc_all.copy()
            z[mm] = np.where(mask, zb[mm], zc_all[mm])   # 제외 행은 CatBoost 없이
            p1 = shift_to_prior(1 / (1 + np.exp(-(z[mm] + ax))), ybar)
            d = bss(yv, p1) - base
            s1 = np.bincount(cl, weights=(p1 - yv) ** 2, minlength=n_id)
            rng = np.random.default_rng(42)
            dr = np.empty(B)
            for b in range(B):
                pk = rng.choice(seen, size=len(seen), replace=True)
                dr[b] = 1e5 * (s0[pk].sum() - s1[pk].sum()) / (n[pk].sum() * ref)
            lo, hi = np.percentile(dr, [2.5, 97.5])
            res[name] = d
            star = "  <<<" if name.startswith("F") else ""
            print(f"  {name:<18s}{expo:>7.1%}{d:>+8.3f}{lo:>+8.3f}{hi:>+8.3f}"
                  f"{np.mean(dr>0):>8.2f}{star}")
        pl = np.array([v for k, v in res.items() if k.startswith("위약")])
        cand = res["F (후보)"]
        print(f"  ── 위약 12개: 평균 {pl.mean():+.3f}  최대 {pl.max():+.3f}  "
              f"최소 {pl.min():+.3f}  sd {pl.std():.3f}")
        z_sc = (cand - pl.mean()) / max(pl.std(), 1e-9)
        print(f"  ── 후보 {cand:+.3f}  →  위약 분포 대비 z = {z_sc:+.2f}"
              f"   ({'최대 초과 ✅' if cand > pl.max() else '**최대 미만 ❌**'})")
    print("\n  기전 판정: fold2024(fspec 있음)에서만 효과가 나야 한다.")
    print("            fold2023 에서도 같이 나오면 'fspec 희석'이 아니라")
    print("            'CatBoost 가 F 행에서 그냥 나쁘다'는 다른 이야기다.")


if __name__ == "__main__":
    main()
