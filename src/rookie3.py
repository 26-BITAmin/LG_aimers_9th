"""§7.78 후속 — R 투수 신인 셀이 fold2023 404 vs fold2024 919로 벌어진 이유.

앞 결론("2023 단독 사건")은 **사후에 지어낸 이야기**였다. 검증한 적이 없다.
여기서 세 가지 후보를 실제로 가른다.

  (a) 노이즈       두 fold 셀의 차이가 클러스터 SE 안인가
  (b) 소수 지배    신인 투수 몇 명이 셀을 끌고 갔는가 (잭나이프 최대 영향)
  (c) 구조         Murphy 분해(RES/REL)와 **투수간/투수내 분해**로
                   "레벨이 틀렸나 / 못 가르나 / 누구를 못 가르나"를 분리

투수간/투수내 분해가 핵심이다. cold-start 가설이 맞다면 신인의 손실은
**투수간**(누가 좋은 투수인지 못 고름)에 몰려야 한다 — 경력 asof가 없으니까.
상황 신호(카운트·주자)는 신인이든 아니든 똑같이 있으므로 투수내는 안 줄어야 한다.

실행:  .venv/Scripts/python.exe -u src/rookie3.py 2>&1 | tee experiments/rookie3.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import TRAIN_CSV
from data import load_csv
from rookie import FOLDS, deployed_pred, prior_seasons
from validation import auc, brier

NBIN = 20
B = 1000
RNG = np.random.default_rng(42)


def murphy(y, p, nbin=NBIN):
    """BS = ref − RES + REL. 분위 빈으로 E[y|p]를 추정한다."""
    q = np.quantile(p, np.linspace(0, 1, nbin + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(p, q[1:-1])
    ybar = y.mean()
    res = rel = 0.0
    for k in range(nbin):
        m = b == k
        if not m.any():
            continue
        w = m.mean()
        ok = y[m].mean()
        res += w * (ok - ybar) ** 2
        rel += w * (p[m].mean() - ok) ** 2
    return res, rel


def decomp(y, p, cl):
    """투수간 / 투수내 분해.

    p_between = 그 투수의 예측 평균 p̄_j  → 투수를 고르는 능력만 남긴 예측
    투수내 이득 = BSS(p) − BSS(p_between)
    ⚠️ p̄_j는 **그 셀 안에서** 계산한다 (셀 밖 행을 끌어오면 다른 물건이 된다).
    """
    ncl = cl.max() + 1
    cnt = np.bincount(cl, minlength=ncl)
    psum = np.bincount(cl, weights=p, minlength=ncl)
    pbar_j = np.divide(psum, cnt, out=np.zeros(ncl), where=cnt > 0)
    ref = y.mean() * (1 - y.mean())
    bss_full = 1e5 * (1 - brier(y, p) / ref)
    bss_btw = 1e5 * (1 - brier(y, pbar_j[cl]) / ref)
    return bss_full, bss_btw, bss_full - bss_btw


def jack(y, p, cl):
    """투수 1명 제거 시 로컬 BSS의 최대 변동 — 소수 지배 여부."""
    ncl = cl.max() + 1
    n = np.bincount(cl, minlength=ncl)
    sq = np.bincount(cl, weights=(p - y) ** 2, minlength=ncl)
    ys = np.bincount(cl, weights=y, minlength=ncl)
    N, S, Y = n.sum(), sq.sum(), ys.sum()
    keep = n > 0
    nn, ss, yy = N - n[keep], S - sq[keep], Y - ys[keep]
    yb = yy / nn
    b = 1e5 * (1 - (ss / nn) / (yb * (1 - yb)))
    base = 1e5 * (1 - (S / N) / ((Y / N) * (1 - Y / N)))
    d = b - base
    j = np.argmax(np.abs(d))
    return int(keep.sum()), float(d[j]), float(n[keep][j] / N * 100)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)

    print(f"{'fold/셀':<16}{'행':>9}{'투수':>6}{'ȳ':>8}{'std(p)':>8}{'AUC':>7}"
          f"{'RES':>9}{'REL':>9}{'로컬BSS':>10}{'투수간':>9}{'투수내':>9}"
          f"{'잭나이프':>10}{'상위1투수%':>10}")
    store = {}
    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pid = va["pitcher_id"].to_numpy()

        for lab, m in (("R 신인", (~gt_f) & (pp == 0)), ("R 2+시즌", (~gt_f) & (pp >= 2)),
                       ("F 신인", gt_f & (pp == 0)), ("F 2+시즌", gt_f & (pp >= 2))):
            yg, pg = y[m], p[m]
            _, cl = np.unique(pid[m], return_inverse=True)
            ref = yg.mean() * (1 - yg.mean())
            res, rel = murphy(yg, pg)
            full, btw, wit = decomp(yg, pg, cl)
            npit, jmax, jshare = jack(yg, pg, cl)
            store[(fold, lab)] = (yg, pg, cl)
            print(f"{str(fold)+' '+lab:<16}{m.sum():>9,}{npit:>6}{yg.mean():>8.4f}"
                  f"{pg.std():>8.4f}{auc(yg, pg):>7.4f}"
                  f"{1e5*res/ref:>9.1f}{1e5*rel/ref:>9.1f}{full:>10.1f}"
                  f"{btw:>9.1f}{wit:>9.1f}{jmax:>+10.1f}{jshare:>10.2f}")

    # fold 간 차이의 유의성 — 두 fold는 서로 다른 행/투수라 비쌍체 클러스터 부트스트랩
    print("\nfold 간 차이 (2024 − 2023), 클러스터 부트스트랩 B=%d" % B)
    for lab in ("R 신인", "R 2+시즌", "F 신인", "F 2+시즌"):
        vals = {}
        for fold in FOLDS:
            yg, pg, cl = store[(fold, lab)]
            ncl = cl.max() + 1
            n = np.bincount(cl, minlength=ncl)
            sq = np.bincount(cl, weights=(pg - yg) ** 2, minlength=ncl)
            ys = np.bincount(cl, weights=yg, minlength=ncl)
            bs = np.empty(B)
            for b in range(B):
                i = RNG.integers(0, ncl, ncl)
                N, S, Y = n[i].sum(), sq[i].sum(), ys[i].sum()
                yb = Y / N
                bs[b] = 1e5 * (1 - (S / N) / (yb * (1 - yb)))
            vals[fold] = (1e5 * (1 - (sq.sum()/n.sum()) / ((ys.sum()/n.sum())*(1-ys.sum()/n.sum()))), bs)
        d = vals[2024][0] - vals[2023][0]
        se = float(np.sqrt(np.nanvar(vals[2024][1], ddof=1) + np.nanvar(vals[2023][1], ddof=1)))
        print(f"  {lab:<10} 2023 {vals[2023][0]:>8.1f}   2024 {vals[2024][0]:>8.1f}"
              f"   Δ {d:>+8.1f} ± {se:>6.1f}  (t {d/se:+.2f})")


if __name__ == "__main__":
    main()
