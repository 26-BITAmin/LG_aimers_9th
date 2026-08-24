"""§7.147 후속 — calibration slope 를 **부분군별로** 쪼갠다 (사용자 지시).\n\n전역 `b = 1.008` 은 "캘리브레이션이 좋다"가 아니라 "전역 노브가 소진됐다"이다.\n§7.81 이 같은 함정을 기록했다 — 전역 T 최적이 1.0 처럼 보인 건 **반대 부호의 두\n오차가 상쇄돼서**였고, 쪼개니 느린 성분 1.37~1.74 / 빠른 성분 0.36~0.55 였다.\n\n**핵심 판정: `(b_g − 1)` 의 부호가 버킷 간에 갈리는가.**\n갈리면 전역 스칼라로 못 고치는 진짜 채널이고, 안 갈리면 전역이 이미 먹은 것이다.\n\n### 축 두 개를 **같은 기계에** 통과시킨다\n\n| 축 | k | 성격 |\n| --- | --- | --- |\n| `asof_pitcher_n` 5분위 | 5 | 후보. §7.81 이 오라클 상한 +47~57 을 찾은 축 |\n| `game_dayofweek` | 7 | **위약.** §7.45 가 이 축으로 없던 +22.8 을 만들었다 |\n\n위약이 `k=7` 로 후보보다 자유도가 **더 많다** — 보수적 영가설이다.\n(저장소 교훈: 셔플은 0 을 주지만 진짜로 죽은 축은 후보와 같은 프로파일을 준다.)\n\n### ⚠️ 분위 경계는 **시즌 ≤ 2022 에서 한 번만** 얼린다\n\nfold 별 train 에서 뽑으면 2023 은 [413, 997, 1843, 3323], 2024 는\n[476, 1168, 2157, 3963] 로 달라진다 (train 이 커지며 커리어 누적이 오른쪽으로\n밀린다). 그러면 "2023 의 3번 버킷"과 "2024 의 3번 버킷"이 **다른 집단**이라\n전이 검정이 성립하지 않는다. 두 fold 공통 부분집합에서 한 번만 얼려 같은 자를 댄다.\n각 행은 자기 `asof_pitcher_n` 하나로 자기 버킷을 안다 — 행 간 집계 없음 (§3.2).\n\n### 적용 시 재-shift 는 **전역 한 번만**\n\n버킷마다 다시 prior 로 옮기면 그건 기울기가 아니라 **버킷별 레벨 보정**이고\n(§7.123 세그먼트 축), 성격이 다른 조작이 섞인다. 여기서는 기울기만 잰다.\n\n실행:  .venv/Scripts/python.exe -u src/calsub.py 2>&1 | tee experiments/calsub.log\n"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import numpy as np

import deploy29 as D
from calslope import (EPS, B_BOOT, build_layers, calib_slope, dbss_boot)
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, shift_to_prior

EDGE_SEASON = min(D.FOLDS)          # 경계를 얼릴 기준: 시즌 < 2023
MIN_ROWS = 2000                     # 이보다 작은 버킷은 b 추정이 무의미
AXES = ("apn5", "dow7")
BAR2 = "\n" + "=" * 100 + "\n"


def make_buckets(df, s, fold, mm, axis, edges=None):
    """행 단위 버킷 코드와 라벨. edges 는 호출자가 얼려서 넘긴다."""
    d = df[s == fold]
    if axis == "apn5":
        v = d["asof_pitcher_n"].to_numpy(dtype=np.float64)[mm]
        b = np.digitize(v, edges)
        lab = ([f"n<{edges[0]:.0f}"]
               + [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
               + [f"n≥{edges[-1]:.0f}"])
        return b, lab
    if axis == "dow7":
        v = d["game_dayofweek"].to_numpy(dtype=np.int64)[mm]
        return v, ["월", "화", "수", "목", "금", "토", "일"]
    raise ValueError(axis)


def subgroup_slope(yv, p, cl, n_id, bucket, labels):
    """부분군별 calibration slope. 각 버킷 안에서 투수 클러스터로 SE 를 잡는다."""
    out = []
    for g, lab in enumerate(labels):
        m = bucket == g
        if m.sum() < MIN_ROWS:
            out.append((lab, int(m.sum()), None))
            continue
        r = calib_slope(yv[m], p[m], groups=cl[m], n_g=n_id,
                        bss_current=bss(yv[m], p[m]))
        out.append((lab, int(m.sum()), r))
    return out


def apply_bucket_slope(z, ybar, bvec, bucket):
    """버킷별 기울기를 곱하고 **전역으로 한 번** prior 에 맞춘다."""
    return shift_to_prior(1.0 / (1.0 + np.exp(-(z * bvec[bucket]))), ybar)


def main():
    df, s, y = D.load_df()
    rng = np.random.default_rng(42)

    edges = np.quantile(df.loc[s < EDGE_SEASON, "asof_pitcher_n"].to_numpy(float),
                        [0.2, 0.4, 0.6, 0.8])
    print(f"asof_pitcher_n 분위 경계 (시즌 < {EDGE_SEASON} 에서 동결): "
          f"{np.round(edges, 0)}\n두 fold 에 같은 자를 댄다.\n")

    S = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        yv, cl, n_id, mm = st["yv"], st["cl"], st["n_id"], st["mm"]
        ax = st["Xax"] @ st["dep"]
        ybar = yv.mean()
        nm, zb = build_layers(df, s, fold, mm)[-1]      # 배포 등가 최상위 층
        p_beta = shift_to_prior(1 / (1 + np.exp(-(zb + ax))), ybar)
        p_dep = shift_to_prior(
            PROB_CENTER + PROB_SCALE * (p_beta - PROB_CENTER), ybar)
        z_dep = np.log(np.clip(p_dep, EPS, 1 - EPS)
                       / (1 - np.clip(p_dep, EPS, 1 - EPS)))
        gl = calib_slope(yv, p_dep, groups=cl, n_g=n_id, bss_current=bss(yv, p_dep))

        print(f"\n{'='*100}\nfold {fold}  층 {nm}  ·  clean {len(yv):,}행  ·  "
              f"전역 b = {gl['b']:.4f} [{gl['b_ci'][0]:.3f}, {gl['b_ci'][1]:.3f}]"
              f"  BSS {bss(yv, p_dep):.1f}\n{'='*100}")

        S[fold] = dict(yv=yv, cl=cl, n_id=n_id, ybar=ybar, z=z_dep, p=p_dep,
                       gl=gl["b"], buck={}, fit={}, lvl={}, layer=nm)

        for axis in AXES:
            bk, lab = make_buckets(df, s, fold, mm, axis,
                                   edges if axis == "apn5" else None)
            tag = "후보" if axis == "apn5" else "위약"
            print(f"\n  ── {axis} ({tag}, k={len(lab)}) ──  "
                  f"p_dep (배포 등가) 기준")
            print(f"    {'버킷':<12s}{'행':>9s}{'투수':>6s}{'b':>8s}"
                  f"{'95% CI (클러스터)':>22s}{'P(b>1)':>9s}{'a_g':>9s}{'BSS_g':>9s}")
            rows = subgroup_slope(yv, p_dep, cl, n_id, bk, lab)
            bvec = np.ones(len(lab))
            for g, (l, n, r) in enumerate(rows):
                if r is None:
                    print(f"    {l:<12s}{n:>9,d}   (행 부족, b=1 고정)")
                    continue
                bvec[g] = r["b"]
                pg = 1.0 - 0.5 * r["p_wald"] if r["b"] > 1 else 0.5 * r["p_wald"]
                z_ = (r["b"] - 1) / r["se_b"]
                from scipy import stats as _s
                pg = float(_s.norm.cdf(z_))
                print(f"    {l:<12s}{n:>9,d}{r['G']:>6d}{r['b']:>8.4f}"
                      f"   [{r['b_ci'][0]:>7.4f}, {r['b_ci'][1]:>7.4f}]{pg:>9.3f}"
                      f"{r['a_large']:>9.4f}{bss(yv, p_dep)*0 + bss(yv[bk==g], p_dep[bk==g]):>9.1f}")
            S[fold]["buck"][axis] = bk
            S[fold]["fit"][axis] = bvec
            avec = np.zeros(len(lab))
            for g, (l, n, r) in enumerate(rows):
                if r is not None:
                    avec[g] = r["a_large"]
            # 가중평균 0 으로 중심화 — 전역 레벨은 안 건드리고 버킷 간 배분만
            # 바꾼다 (§7.123 과 같은 구조).
            w = np.array([(bk == g).sum() for g in range(len(lab))], float)
            avec -= (avec * w).sum() / w.sum()
            S[fold]["lvl"][axis] = avec

            # 부호 갈림 요약 — 이것이 핵심 판정
            ok = [r for _, _, r in rows if r is not None]
            hi = sum(1 for r in ok if r["b_ci"][0] > 1)
            lo = sum(1 for r in ok if r["b_ci"][1] < 1)
            print(f"    → CI 가 1 위: {hi}개 · 1 아래: {lo}개 · 0 포함: "
                  f"{len(ok)-hi-lo}개   "
                  f"b 범위 [{min(r['b'] for r in ok):.3f}, "
                  f"{max(r['b'] for r in ok):.3f}]"
                  + ("   ⚠️ 부호 갈림" if hi and lo else "   부호 일치"))

            # in-sample 상한 (같은 fold 에서 적합해 그 fold 에 적용)
            d_in = bss(yv, apply_bucket_slope(z_dep, ybar, bvec, bk)) - bss(yv, p_dep)
            print(f"    in-sample 오라클 상한 ΔBSS {d_in:+.2f}")

    # ── 전이 — 다른 fold 에서 적합한 b_g 를 버킷별로 가져다 쓴다 ──────
    print(f"\n\n{'='*100}\n전이 — b_g 를 **다른 fold** 에서 적합해 적용 "
          f"(클러스터 부트스트랩 B={B_BOOT})\n{'='*100}")
    print(f"  {'fold':<6s}{'축':<8s}{'출처':<8s}{'ΔBSS':>9s}{'2.5%':>9s}"
          f"{'97.5%':>9s}{'P(Δ>0)':>9s}   비고")
    for fold in D.FOLDS:
        e = S[fold]
        other = [f for f in D.FOLDS if f != fold][0]
        # 대조군 ①: 전역 b 하나만 (§7.147 이 이미 낸 값)
        pg = shift_to_prior(1 / (1 + np.exp(-(e["z"] * S[other]["gl"]))), e["ybar"])
        d, lo, hi, pp = dbss_boot(e["yv"], e["p"], pg, e["cl"], e["n_id"], rng)
        print(f"  {fold:<6d}{'전역':<8s}{other:<8d}{d:>9.2f}{lo:>9.2f}{hi:>9.2f}"
              f"{pp:>9.3f}   b={S[other]['gl']:.4f}")
        for axis in AXES:
            bvec = S[other]["fit"][axis]
            p1 = apply_bucket_slope(e["z"], e["ybar"], bvec, e["buck"][axis])
            d, lo, hi, pp = dbss_boot(e["yv"], e["p"], p1, e["cl"], e["n_id"], rng)
            tag = "후보" if axis == "apn5" else "위약"
            print(f"  {fold:<6d}{axis:<8s}{other:<8d}{d:>9.2f}{lo:>9.2f}{hi:>9.2f}"
                  f"{pp:>9.3f}   {tag}")

    # ── 버킷 계수가 fold 를 건너 **지속**하는가 ────────────────────
    print(f"\n{'='*100}\n버킷 계수의 fold 간 상관 — 구조가 지속하는가"
          f"\n{'='*100}")
    f0, f1 = D.FOLDS
    print(f"  {'축':<8s}{'corr(b_g)':>12s}{'corr(a_g)':>12s}"
          f"{'a_g 부호 일치':>14s}")
    for axis in AXES:
        b0, b1 = S[f0]["fit"][axis], S[f1]["fit"][axis]
        a0, a1 = S[f0]["lvl"][axis], S[f1]["lvl"][axis]
        agree = int(np.sum(np.sign(a0) == np.sign(a1)))
        print(f"  {axis:<8s}{np.corrcoef(b0, b1)[0,1]:>12.3f}"
              f"{np.corrcoef(a0, a1)[0,1]:>12.3f}{agree:>10d}/{len(a0)}")

    # ── 레벨(a_g) 전이 — 기울기와 **같은 기계**, 같은 위약 ────────────
    print(f"\n{'='*100}\n레벨 전이 — a_g 를 **다른 fold** 에서 적합해 버킷별 "
          f"로짓 가산 (B={B_BOOT})\n{'='*100}")
    print(f"  {'fold':<6s}{'축':<8s}{'출처':<8s}{'ΔBSS':>9s}{'2.5%':>9s}"
          f"{'97.5%':>9s}{'P(Δ>0)':>9s}   비고")
    for fold in D.FOLDS:
        e = S[fold]
        other = [f for f in D.FOLDS if f != fold][0]
        for axis in AXES:
            avec = S[other]["lvl"][axis]
            p1 = shift_to_prior(
                1 / (1 + np.exp(-(e["z"] + avec[e["buck"][axis]]))), e["ybar"])
            d, lo, hi, pp = dbss_boot(e["yv"], e["p"], p1, e["cl"], e["n_id"], rng)
            tag = "후보" if axis == "apn5" else "위약"
            print(f"  {fold:<6d}{axis:<8s}{other:<8d}{d:>9.2f}{lo:>9.2f}"
                  f"{hi:>9.2f}{pp:>9.3f}   {tag}")
        # in-sample 상한
        for axis in AXES:
            avec = e["lvl"][axis]
            p1 = shift_to_prior(
                1 / (1 + np.exp(-(e["z"] + avec[e["buck"][axis]]))), e["ybar"])
            print(f"  {fold:<6d}{axis:<8s}{'(자기)':<8s}"
                  f"{bss(e['yv'], p1) - bss(e['yv'], e['p']):>9.2f}"
                  f"{'':>27s}   in-sample 상한")

    # ── κ 수축 스윕 — corr 0.979 인데 전이가 갈리는 이유는 **크기**다 ──
    # 신인 버킷 a_g 가 2023 -0.0808 -> 2024 -0.0397 로 반감한다. 관계는 지속하고
    # 크기가 준다. 그러면 처방은 기각이 아니라 kappa < 1 이다 (7.37 밴드 규칙).
    print(BAR2 + "kappa 수축 스윕 — kappa*a_g(타 fold) 를 적용" + BAR2)
    KAP = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25]
    for axis in AXES:
        tag = "후보" if axis == "apn5" else "위약"
        print("  " + axis + " (" + tag + ")")
        head = "    " + "kappa".rjust(7)
        for fold in D.FOLDS:
            head += ("f" + str(fold)).rjust(11)
        print(head + "크기비".rjust(9))
        rows = []
        for k in KAP:
            ds = []
            for fold in D.FOLDS:
                e = S[fold]
                other = [f for f in D.FOLDS if f != fold][0]
                avec = k * S[other]["lvl"][axis]
                p1 = shift_to_prior(
                    1 / (1 + np.exp(-(e["z"] + avec[e["buck"][axis]]))), e["ybar"])
                ds.append(bss(e["yv"], p1) - bss(e["yv"], e["p"]))
            lo_, hi_ = min(abs(d) for d in ds), max(abs(d) for d in ds)
            ratio = hi_ / lo_ if lo_ > 1e-9 else float("inf")
            rows.append((k, ds, ratio))
            line = "    " + f"{k:7.3f}"
            for d in ds:
                line += f"{d:>11.2f}"
            both = all(d > 0 for d in ds)
            print(line + f"{ratio:>9.2f}" + ("   양 fold 양수" if both else ""))




if __name__ == "__main__":
    main()
