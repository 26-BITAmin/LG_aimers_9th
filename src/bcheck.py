"""부트스트랩 반복수 B 가 판정을 바꾸는지 확인. 학습 0회.

병행 세션 §7.143 지적: B=400 에서 게이트 판정이 뽑기마다 갈렸다.
내 하네스는 **B=200** 이고, 판정이 걸린 두 칸의 구간 상한이 −0.9 / −0.2 로
0 에 붙어 있다. 몬테카를로 오차가 그 폭보다 크면 판정이 난수에 달려 있다.

가속: shift_to_prior 가 이분법 47회라 B 를 못 올린다 → **Newton 5회**로 바꾸고
로짓을 미리 갖고 있는다 (재로짓 생략). 점추정으로 원 하네스와 대조한다.
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
from config import MLP_ALPHA, ROOT

CACHE = ROOT / "experiments" / "cache"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def bs_shift(z, y, ybar, iters=6):
    """logit 평행이동으로 mean(p)==ybar 를 맞춘 뒤 Brier 반환 (Newton)."""
    d = 0.0
    for _ in range(iters):
        s = _sig(z + d)
        f = s.mean() - ybar
        g = (s * (1.0 - s)).mean()
        if g < 1e-12:
            break
        d -= f / g
    s = _sig(z + d)
    return ((s - y) ** 2).mean()


def bss_pair(z1, z0, y):
    yb = y.mean()
    ref = yb * (1.0 - yb)
    return -1e5 * (bs_shift(z1, y, yb) - bs_shift(z0, y, yb)) / ref


def main():
    df, s, y = D.load_df()
    import os
    only = os.environ.get("BCHECK_ONLY", "")
    CELLS = [(2023, "lag5", "A", "ctr7"), (2023, "pow5_3", "A", "ctr9"),
             (2024, "pow5_3", "pow5_2", "ctr9")]
    if only == "crit":                      # 판정이 걸린 칸만 (상한 −0.2)
        CELLS = [(2024, "pow5_3", "pow5_2", "ctr9")]
    print(f"{'fold':>5} {'대비':<20}{'점추정':>9}   B=200 (3뽑기)                B=4000 (3뽑기)")
    for fold, a, b, pre in CELLS:
        st = D.state(df, s, y, fold)
        mm, yv, cl, n_id = st["mm"], st["yv"], st["cl"], st["n_id"]
        Xax, dep = st["Xax"], st["dep"]

        def zof(name, prefix):
            if name == "A":
                return st["z0"]
            with np.load(CACHE / f"{prefix}_fold{fold}_{name}.npz") as d:
                z = (1 - MLP_ALPHA) * d["z_lgb"] + MLP_ALPHA * d["z_mlp"]
            return z[mm] + Xax @ dep

        z1, z0 = zof(a, pre), zof(b, "ctr9" if b == "pow5_2" else pre)
        pt = bss_pair(z1, z0, yv)

        order = np.argsort(cl, kind="stable")
        bnd = np.searchsorted(cl[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        groups = [order[bnd[j]:bnd[j + 1]] for j in range(n_id) if bnd[j + 1] > bnd[j]]

        def run(B, seed):
            rng = np.random.default_rng(seed)
            out = np.empty(B)
            for i in range(B):
                pick = rng.integers(0, len(groups), len(groups))
                idx = np.concatenate([groups[j] for j in pick])
                out[i] = bss_pair(z1[idx], z0[idx], yv[idx])
            lo, hi = np.percentile(out, [2.5, 97.5])
            return lo, hi

        c200 = "  ".join(f"[{lo:+5.1f},{hi:+5.1f}]" for lo, hi in
                         (run(200, sd) for sd in (1, 2, 3)))
        c4k = "  ".join(f"[{lo:+5.1f},{hi:+5.1f}]" for lo, hi in
                        (run(4000, sd) for sd in (1, 2, 3)))
        print(f"{fold:>5} {f'{a} - {b}':<20}{pt:>+9.1f}   {c200}   {c4k}")


if __name__ == "__main__":
    main()
