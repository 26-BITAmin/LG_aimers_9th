"""§7.168 — F−R BSS 격차의 분해: 구성(이력 얇은 행 비중) vs 같은 칸 내 성능.

사용자 질문: "F가 R보다 잘 안나오는데 어느 지점에서? F에 신입이 훨씬 많아서인가?"

방법
  리그 G 의 BSS 는 그룹-로컬 기준선으로 채점: BSS_G = 1e5·(1 − BS_G / ȳ_G(1−ȳ_G)).
  통산 기록 버킷 c 로 쪼개면 BS_G = Σ_c w_c·ref_c·(1−skill_c)  (항등식 — assert 확인).
  반사실 CF1(F구성 + R실력) = F 의 (w, ref) 에 R 의 칸별 skill 을 이식.
    경로 A: gap = [R − CF1](구성 몫) + [CF1 − F](칸 내 성능 몫)
    경로 B: CF2(R구성 + F실력) 로 반대 방향. 두 경로 차이(상호작용)도 보고한다
    (memory: 소거 사슬은 증거가 아니다 — 경로를 바꿔 같은 항을 다시 잰다).
  성분별 95% CI 는 투수 클러스터 부트스트랩 B=1000.

버킷은 저표본 칸 병합판: 0~99 / 100~499 / 500~1,999 / 2,000+  (F 10,000+ 이 n=8~42 라
fine 버킷은 부트스트랩에서 빈 칸이 생긴다). 코호트(신인 pp=0 / 2년차 pp=1 / 3+년)로도
같은 분해를 반복한다 — 사용자 가설("신입 비중")을 직접 겨눈다.

실행:  .venv/Scripts/python.exe -u src/rookdiag7.py 2>&1 | tee experiments/rookdiag7.log
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

import rookdiag as R
from config import TRAIN_CSV
from data import load_csv
from validation import brier

B = 1000
RNG = np.random.default_rng(7)
EDGES = [100, 500, 2000]
LABS = ["0~99", "100~499", "500~1,999", "2,000+"]


def cellsums(cl, n_id, y, p, mask):
    """칸 하나의 투수별 (행수, Σ(p−y)², Σy)."""
    idx = cl[mask]
    n = np.bincount(idx, minlength=n_id).astype(np.float64)
    se = np.bincount(idx, weights=(p[mask] - y[mask]) ** 2, minlength=n_id)
    sy = np.bincount(idx, weights=y[mask], minlength=n_id)
    return n, se, sy


def agg(sums, k):
    """draw k (None=원표본)에서 칸별 (N, BS합, y합)."""
    out = []
    for n, se, sy in sums:
        if k is None:
            out.append((n.sum(), se.sum(), sy.sum()))
        else:
            out.append((n[k].sum(), se[k].sum(), sy[k].sum()))
    return out


def decompose(aF, aR):
    """칸별 집계 → 리그 BSS·반사실·성분. 실패(빈 칸/퇴화 ref)는 None."""
    NF = sum(a[0] for a in aF)
    NR = sum(a[0] for a in aR)
    if NF == 0 or NR == 0:
        return None
    yF = sum(a[2] for a in aF) / NF
    yR = sum(a[2] for a in aR) / NR
    refF, refR = yF * (1 - yF), yR * (1 - yR)
    bsF = sum(a[1] for a in aF) / NF
    bsR = sum(a[1] for a in aR) / NR
    bssF = 1e5 * (1 - bsF / refF)
    bssR = 1e5 * (1 - bsR / refR)

    def cell(a):
        N, S, Y = a
        if N == 0:
            return None
        yb = Y / N
        ref = yb * (1 - yb)
        if ref <= 0:
            return None
        return N, ref, 1 - (S / N) / ref          # N, ref_c, skill_c

    cF = [cell(a) for a in aF]
    cR = [cell(a) for a in aR]
    if any(c is None for c in cF + cR):
        return None
    cf1 = sum(N / NF * ref * (1 - skR) for (N, ref, _), (_, _, skR) in zip(cF, cR))
    cf2 = sum(N / NR * ref * (1 - skF) for (N, ref, _), (_, _, skF) in zip(cR, cF))
    bss_cf1 = 1e5 * (1 - cf1 / refF)              # F구성 + R실력
    bss_cf2 = 1e5 * (1 - cf2 / refR)              # R구성 + F실력
    gap = bssR - bssF
    return dict(
        bssF=bssF, bssR=bssR, gap=gap,
        compA=bssR - bss_cf1, withinA=bss_cf1 - bssF,
        withinB=bssR - bss_cf2, compB=bss_cf2 - bssF,
        skdiff=[cR[i][2] * 1e5 - cF[i][2] * 1e5 for i in range(len(cF))],
        contrib=[1e5 * cF[i][0] / NF * cF[i][1] * (cR[i][2] - cF[i][2]) / refF
                 for i in range(len(cF))],
        skF=[c[2] * 1e5 for c in cF], skR=[c[2] * 1e5 for c in cR],
    )


def boot(sumsF, sumsR, n_id):
    keys = ["gap", "compA", "withinA", "withinB", "compB"]
    out = {k: [] for k in keys}
    sk = [[] for _ in sumsF]
    for _ in range(B):
        k = RNG.integers(0, n_id, n_id)
        d = decompose(agg(sumsF, k), agg(sumsR, k))
        if d is None:
            continue
        for key in keys:
            out[key].append(d[key])
        for i, v in enumerate(d["skdiff"]):
            sk[i].append(v)
    ci = {k: np.percentile(out[k], [2.5, 97.5]) for k in keys}
    ci["skdiff"] = [np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
                    for v in sk]
    ci["ndraw"] = len(out["gap"])
    return ci


def fmt_ci(ci):
    return f"[{ci[0]:+7.1f}, {ci[1]:+7.1f}]"


def run_partition(title, labs, masksF, masksR, cl, n_id, y, p, isF):
    sumsF = [cellsums(cl, n_id, y, p, m) for m in masksF]
    sumsR = [cellsums(cl, n_id, y, p, m) for m in masksR]
    d = decompose(agg(sumsF, None), agg(sumsR, None))
    ci = boot(sumsF, sumsR, n_id)

    # 항등식: 칸 재조립 BSS == 직접 BSS
    for m_all, name, val in ((isF, "F", d["bssF"]), (~isF, "R", d["bssR"])):
        yg = y[m_all].mean()
        direct = 1e5 * (1 - brier(y[m_all], p[m_all]) / (yg * (1 - yg)))
        assert abs(direct - val) < 1e-6, (name, direct, val)

    print(f"\n  ── {title} ──")
    print(f"    리그-로컬 BSS   R {d['bssR']:7.1f}   F {d['bssF']:7.1f}   "
          f"격차(R−F) {d['gap']:+7.1f}  {fmt_ci(ci['gap'])}")
    print(f"\n    {'칸':<12}{'F행비중':>8}{'R행비중':>8}{'skill_F':>9}{'skill_R':>9}"
          f"{'차(R−F)':>9}{'95% CI':>20}{'격차기여':>9}")
    nF = [m.sum() for m in masksF]
    nR = [m.sum() for m in masksR]
    for i, lab in enumerate(labs):
        print(f"    {lab:<12}{nF[i]/sum(nF)*100:>7.1f}%{nR[i]/sum(nR)*100:>7.1f}%"
              f"{d['skF'][i]:>9.1f}{d['skR'][i]:>9.1f}{d['skdiff'][i]:>+9.1f}"
              f"{fmt_ci(ci['skdiff'][i]):>20}{d['contrib'][i]:>+9.1f}")
    print(f"      (격차기여 = F구성 가중으로 잰 칸별 성능 몫. 합 = 경로A 칸내성능 몫)")
    print(f"\n    분해 (격차 {d['gap']:+.1f} 중)")
    print(f"      경로 A   구성 몫 {d['compA']:+7.1f} {fmt_ci(ci['compA'])}"
          f"   칸내 성능 몫 {d['withinA']:+7.1f} {fmt_ci(ci['withinA'])}")
    print(f"      경로 B   구성 몫 {d['compB']:+7.1f} {fmt_ci(ci['compB'])}"
          f"   칸내 성능 몫 {d['withinB']:+7.1f} {fmt_ci(ci['withinB'])}")
    print(f"      상호작용 (경로차) {d['compA']-d['compB']:+7.1f}"
          f"   (부트스트랩 유효 draw {ci['ndraw']}/{B})")
    return d


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    for fold in (2024, 2023):
        va, y, p, _ = R.build(fold, df, s)
        an = va["asof_pitcher_n"].to_numpy(dtype=np.float64)
        isF = va["game_type"].to_numpy().astype(str) == "F"
        pp = R.prior_seasons(df, s, fold, "pitcher_id")
        cl, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        bkt = np.digitize(np.nan_to_num(an, nan=0.0), EDGES)
        tag = "★ 주 판단" if fold == 2024 else "참고 (2019~22 F 라벨 오염 시즌 학습)"
        print("\n" + "=" * 100)
        print(f"fold {fold}   n={len(y):,}   F행 {isF.mean()*100:.1f}%   — {tag}")
        print("=" * 100)

        # 구성 요약 — 사용자 가설 검사용
        print(f"\n  구성 요약 (리그 내부 비중)")
        for name, m in (("R", ~isF), ("F", isF)):
            rk = pp[m] == 0
            print(f"    {name}: 신인 {rk.mean()*100:5.1f}%   통산<500 "
                  f"{(an[m] < 500).mean()*100:5.1f}%   신인 중앙 asof_n "
                  f"{np.nanmedian(an[m][rk]):6,.0f}   전체 중앙 asof_n "
                  f"{np.nanmedian(an[m]):6,.0f}")

        run_partition("통산 기록 버킷 분해", LABS,
                      [isF & (bkt == i) for i in range(len(LABS))],
                      [(~isF) & (bkt == i) for i in range(len(LABS))],
                      cl, n_id, y, p, isF)
        run_partition("코호트(신인/2년차/3+) 분해", ["신인 pp=0", "2년차 pp=1", "3+ pp≥2"],
                      [isF & (pp == 0), isF & (pp == 1), isF & (pp >= 2)],
                      [(~isF) & (pp == 0), (~isF) & (pp == 1), (~isF) & (pp >= 2)],
                      cl, n_id, y, p, isF)


if __name__ == "__main__":
    main()
