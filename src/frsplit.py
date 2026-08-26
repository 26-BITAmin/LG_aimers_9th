"""기각 후보들의 ΔBSS 를 game_type(F/R) 으로 분해. 학습 0회, 캐시만 읽는다.

병행 세션 §7.142 제안: 효과가 F 행에 몰려 있으면 s35 층(fspec)과 상호작용해
부호까지 바뀔 수 있다. 노출 비율 근처면 균등 분산 → s35 층은 축소만 한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"D:\projects\LG aimers 야구") / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import deploy29 as D
import foldcache as _fc
from config import MLP_ALPHA, ROOT
from validation import shift_to_prior

KEY = _fc._key()
CACHE = ROOT / "experiments" / "cache"
ARMS = [("ctr7", "lag5mu"), ("ctr7", "mu5"), ("ctr7", "mu5dead"), ("ctr7", "lag5"),
        ("ctr8", "lag5sq"), ("ctr9", "pow5_2"), ("ctr9", "pow5_3")]

df, s, y = D.load_df()
print(f"{'팔':<10}{'fold':>6}{'ΔBSS':>9}{'F기여':>9}{'F기여율':>9}{'F노출율':>9}{'배율':>8}")
for fold in (2023, 2024):
    st = D.state(df, s, y, fold)
    mm, yv, z0, p0 = st["mm"], st["yv"], st["z0"], st["p0"]
    Xax, dep = st["Xax"], st["dep"]
    gt = st["d"]["game_type"].astype(str).to_numpy()
    isF = gt == "F"
    expo = isF.mean()
    ybar = yv.mean()
    bsref = ybar * (1 - ybar)
    n = len(yv)
    for pre, m in ARMS:
        f = CACHE / f"{pre}_fold{fold}_{m}.npz"
        if not f.exists():
            continue
        with np.load(f) as d:
            z = (1 - MLP_ALPHA) * d["z_lgb"] + MLP_ALPHA * d["z_mlp"]
        p1 = shift_to_prior(1 / (1 + np.exp(-(z[mm] + Xax @ dep))), ybar)
        di = (p1 - yv) ** 2 - (p0 - yv) ** 2          # 행별 BS 증가분
        dbss = -1e5 * di.sum() / n / bsref            # 총 ΔBSS
        dF = -1e5 * di[isF].sum() / n / bsref         # F 행 기여
        share = dF / dbss if abs(dbss) > 1e-9 else float("nan")
        print(f"{m:<10}{fold:>6}{dbss:>9.1f}{dF:>9.1f}{share:>9.2f}{expo:>9.2f}"
              f"{share/expo if expo else float('nan'):>8.2f}")
    print()
print("배율 ≈ 1 이면 F/R 에 균등 분산 (노출 비율만큼만 기여) → s35 층은 축소만 한다.")
print("배율 ≫ 1 이면 F 행에 몰림 → fspec 과 자리를 다툴 수 있어 재검증 값어치 있음.")
