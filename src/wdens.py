"""`weight_decay` 이득이 **앙상블에서도 살아남는가** (§7.174 후속).

§7.174 는 복제 단위가 **단일 emb4 멤버**였고 `wd 0.01→0.03` 이 +0.42 (t=4.38, 7/7) 였다.
그런데 배포 T1 의 MLP 는 **8멤버 앙상블**이다. `wd` 를 키우는 것은 가중치 수축 =
**분산 축소**이고 앙상블 평균도 분산 축소다. 두 수단이 겹치면 단일 멤버에서 잰 이득이
앙상블에서 사라진다 — 이미 데인 적 있는 함정이다.

**설계.** 멤버를 5개씩 학습해두고 **같은 멤버로** M=1,3,5 부분앙상블을 만들어 채점한다.
추가 학습 비용 0 으로 Δ(M) 곡선이 나온다. 겹침이 있으면 M 이 커질수록 Δ 가 준다.

  · 복제 = 7 (§7.172~174 와 동일)
  · 팔 = 기준(wd .01) / wd .03
  · M=1 이 §7.174 의 +0.42 를 재현해야 한다 — **알려진 수치 재현이 먼저다.**
    재현이 안 되면 하네스가 달라진 것이고 M 곡선은 읽을 수 없다.

z_lgb 는 `mlpreg.py` 캐시를 그대로 쓴다 (팔 사이 고정, 쌍체 차분에서 상쇄).

실행:  .venv/Scripts/python.exe -u src/wdens.py 2024
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
import mlp
import modelabel
import soften
from config import MLP_ALPHA, MLP_AUX_LAMBDA, TRAIN_CSV
from data import load_csv

mlp.THREADS = 8
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
REPS = (42, 7, 2024, 1234, 31337, 777, 31415)     # 복제 7
M_LIST = (1, 3, 5)
NMEM = 5
EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
BASE = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, use_id_emb=True, emb_dim=EMB4)
ARMS = {"기준 wd.01": 1e-2, "wd .03": 3e-2}


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    del df; gc.collect()
    aux0, cov = modelabel.aux_labels(tr)
    X, y, w, aux, _ = soften.soften(tr, Xtr0, ytr0, w0, aux0)
    z_lgb = np.load(CACHE / f"mlpreg_lgb_f{fold}.npz")["z"]
    print(f"[{time.time()-t0:.0f}s] fold={fold}  soft {len(Xtr0):,}→{len(X):,}"
          f"  보조라벨 {cov:.1%}  LGBM 재사용 단독 {mlp.score(z_lgb, yva):.1f}")
    A, C, B, Cv, _, sizes = mlp.encode(X, Xva, art)
    del Xtr0, X; gc.collect()

    # 캐시 키 유일성 — §7.173 에서 "l2 10"/"l2 100" 이 같은 파일을 가리켰다
    import re
    slugs = {n: re.sub(r"[^\w.]+", "_", n).strip("_") for n in ARMS}
    assert len(set(slugs.values())) == len(ARMS), f"캐시 키 충돌: {slugs}"

    S = {}                                   # S[arm][M] = 복제 7개 점수
    for arm, wd in ARMS.items():
        rows = {M: [] for M in M_LIST}
        for r in REPS:
            f = CACHE / f"wdens_f{fold}_{slugs[arm]}_r{r}.npz"
            if f.exists():
                Z = np.load(f)["Z"]
            else:
                Z = np.stack([
                    mlp.train_mlp(A, C, y, w, B, Cv, sizes, seed=r * 10 + i,
                                  verbose=False, y_aux=aux,
                                  **dict(BASE, wd=wd, aux=MLP_AUX_LAMBDA))[1][-1]
                    for i in range(NMEM)])
                np.savez_compressed(f, Z=Z)
                gc.collect()
            for M in M_LIST:
                zm = Z[:M].mean(axis=0)
                rows[M].append(mlp.score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm, yva))
            print(f"[{time.time()-t0:.0f}s] {arm:<11} r={r:<6} "
                  + "  ".join(f"M{M}={rows[M][-1]:.2f}" for M in M_LIST))
        S[arm] = {M: np.array(v) for M, v in rows.items()}

    print(f"\n{'M':>3}{'기준':>10}{'wd .03':>10}{'쌍체Δ':>9}{'SE':>7}{'t':>8}{'양수':>7}{'ρ':>7}  95% CI")
    print("-" * 78)
    for M in M_LIST:
        a, b = S["기준 wd.01"][M], S["wd .03"][M]
        d = b - a
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0.0
        rho = np.corrcoef(a, b)[0, 1]
        print(f"{M:>3}{a.mean():>10.2f}{b.mean():>10.2f}{d.mean():>9.2f}{se:>7.2f}"
              f"{t:>8.2f}{f'{(d>0).sum()}/7':>7}{rho:>7.2f}"
              f"  [{d.mean()-2.447*se:+.2f}, {d.mean()+2.447*se:+.2f}]")
    d1 = (S["wd .03"][1] - S["기준 wd.01"][1]).mean()
    d5 = (S["wd .03"][5] - S["기준 wd.01"][5]).mean()
    print(f"\n겹침 진단:  Δ(M=1) = {d1:+.2f}  →  Δ(M=5) = {d5:+.2f}"
          f"   잔존율 {d5/d1:.0%}" if abs(d1) > 1e-9 else "")
    print("※ M=1 이 §7.174 의 +0.42 를 재현해야 M 곡선을 읽을 수 있다.")
    np.savez_compressed(CACHE / f"wdens_f{fold}.npz",
                        **{f"{slugs[a]}_M{M}": S[a][M] for a in ARMS for M in M_LIST})
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
