"""§7.194c — TabM 의 fold2023 우위가 **오염 노출** 프로파일을 따르는가 (관찰 분석)

⚠️ §7.191~193 은 병렬 세션이 선점 — hillclimb.py 머리말의 재번호 사유 참조.

사용자 가설: fold2023 모델은 오염 시즌(2022, h≈0.30)이 recency 가중 1.0 으로
학습 질량을 지배하고, fold2024 는 깨끗한 2023 이 지배한다 (오염 가중 ≤0.25).
→ fold2023 의 멤버 순위(TabM 79% 선택)는 "시즌 단절"이 아니라
  **잔여 라벨 오염을 아키텍처가 다르게 흡수한 차이**일 수 있다.

예측: 오염 내성 가설이 맞으면 fold2023 에서 TabM−B1 우위가
  투수의 pre-2023 F 노출이 큰 버킷에 집중된다. fold2024 는 노출 자체가
  4배 약해 프로파일이 눕는다.
⚠️ 비대칭 증거다 — 오염은 공유 피처로 무노출 행에도 번지므로
  ([[exposure-diagnostic-only-closes-exposure-channel]]) 평평해도 반증이 아니고,
  기울면 지지다. 판정이 아니라 기전 독해용.

읽기 전용 — 캐시를 쓰지 않는다 (다른 세션의 hillclimb2 와 안전 공존).

실행:  .venv/Scripts/python.exe -u src/tabmexpo.py 2>&1 | tee experiments/tabmexpo.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

import deploy29 as D
import hillclimb as H
from validation import shift_to_prior

CACHE = D.CACHE
FOLDS = (2023, 2024)


def tabm_family(fold, n_expect):
    """hillclimb2 의 'tabm' 가족과 같은 정의: p5*(위약)·ctrl* 제외, 양 fold 짝 필수."""
    tags = set()
    for p in CACHE.glob("tabm_f2023_*.npz"):
        tags.add(p.name[len("tabm_f2023_"):-len(".npz")])
    zs = []
    for t in sorted(tags):
        if t.startswith(("p5", "ctrl")):
            continue
        p23, p24 = CACHE / f"tabm_f2023_{t}.npz", CACHE / f"tabm_f2024_{t}.npz"
        if not (p23.exists() and p24.exists()):
            continue
        z = np.load(CACHE / f"tabm_f{fold}_{t}.npz")["z"]
        assert len(z) == n_expect, t
        zs.append(np.asarray(z, np.float64))
    return np.mean(zs, axis=0), len(zs)


def bucket_table(st, d_i, expo_row, isF_row, label):
    """d_i = (B1 오차) − (TabM 오차) 행별. 버킷별 ΔBSS 환산 + 투수 클러스터 t.
    expo 는 투수 단위라 클러스터가 버킷을 가로지르지 않는다."""
    ref = st["ybar"] * (1 - st["ybar"])
    edges = [-1e-9, 1e-9, 0.05, 0.15, 1.01]      # 0 / (0,5%] / (5,15%] / >15%
    names = ["노출 0", "0~5%", "5~15%", ">15%"]
    rows = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (expo_row > lo) & (expo_row <= hi)
        if not m.any():
            rows.append((nm, 0, np.nan, np.nan))
            continue
        dbss = 1e5 * d_i[m].sum() / (m.sum() * ref)
        # 클러스터 t (버킷 안 투수 단위)
        cl = st["cl"][m]
        g = np.bincount(cl, weights=d_i[m])
        n = np.bincount(cl)
        seen = n > 0
        mu = g[seen].sum() / n[seen].sum()
        se = np.sqrt(np.sum((g[seen] - mu * n[seen]) ** 2)) / n[seen].sum()
        rows.append((nm, int(m.sum()), dbss, mu / se if se > 0 else 0.0))
    print(f"  {label}")
    print(f"    {'버킷':<8s}{'행수':>9s}{'ΔBSS(TabM−B1)':>16s}{'클러스터 t':>12s}")
    for nm, n, dbss, t in rows:
        print(f"    {nm:<8s}{n:>9,d}{dbss:>16.2f}{t:>12.2f}")
    for nm, m in (("F 행", isF_row), ("R 행", ~isF_row)):
        dbss = 1e5 * d_i[m].sum() / (m.sum() * ref)
        print(f"    {nm:<8s}{int(m.sum()):>9,d}{dbss:>16.2f}")


def main():
    df, s, y = D.load_df()
    gt = df["game_type"].to_numpy().astype(str)
    pid = df["pitcher_id"].to_numpy()

    # 투수별 pre-2023 F 노출 = (pre-2023 F 행수) / (pre-2023 전체 행수)
    pre = s < 2023
    codes, uniq = pd.factorize(pid)
    nF = np.bincount(codes[pre & (gt == "F")], minlength=len(uniq))
    nA = np.bincount(codes[pre], minlength=len(uniq))
    expo = np.where(nA > 0, nF / np.maximum(nA, 1), 0.0)
    print(f"투수 {len(uniq)}명   pre-2023 이력 보유 {int((nA>0).sum())}명   "
          f"노출>0 {int(((nA>0)&(nF>0)).sum())}명")

    for fold in FOLDS:
        st0 = D.state(df, s, y, fold)
        st = dict(mask=st0["mm"], ax=st0["Xax"] @ st0["dep"], yv=st0["yv"],
                  ybar=st0["yv"].mean(), cl=st0["cl"], n_id=st0["n_id"])
        import time
        Z, names, fam = H.load_library(df, fold, time.time())
        p1, _ = H.eval_w(st, Z, H.baseline_w(fam, "B1"))       # B1 (배포 등가)
        # TabM 가족 평균 (hillclimb2 진단과 동일 층: +ax, prior shift)
        zt, ntm = tabm_family(fold, len(st["mask"]))
        pt = shift_to_prior(1 / (1 + np.exp(-(zt[st["mask"]] + st["ax"]))), st["ybar"])
        from validation import bss
        print(f"\nfold{fold}  B1 {bss(st['yv'], p1):.1f}   TabM({ntm}) {bss(st['yv'], pt):.1f}")

        d_i = (p1 - st["yv"]) ** 2 - (pt - st["yv"]) ** 2      # >0 이면 TabM 우위
        vi = st0["vi"]
        expo_row = expo[codes[vi][st0["mm"]]]
        isF_row = (gt[vi][st0["mm"]] == "F")
        bucket_table(st, d_i, expo_row, isF_row,
                     f"TabM−B1 우위의 오염 노출 프로파일 (fold{fold})")


if __name__ == "__main__":
    main()
