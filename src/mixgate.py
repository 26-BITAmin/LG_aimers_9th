"""§7.118-D — §7.77 이 남긴 미학습 구조 2축의 3중 무학습 게이트.

대상 (stepprobe, s23 스택에서 측정된 값)
    asof_pitcher_breaking_rate_sm   z 7.5/7.1
    asof_pitcher_fastball_rate      z 5.9/5.7
둘 다 "LGBM·MLP 둘 다 못 배운 구조"로 기록된 채, 믹스 가계 전과(r3_ent LB −11.01,
pitchfine)만을 근거로 미판정으로 남았다. "전적은 근거가 아니다" — 직접 잰다.

3중 게이트 (순서대로, 하나라도 죽으면 종료. 전부 학습 0회)
    A  s29 배포 층 잔차에서 excess z 재현되는가 (stepprobe 동일 통계 + 셔플 널)
       — tm_relh_std 가 현 스택에서 죽은 전례상 **여기서 죽을 확률이 가장 높다**
    B  초과 프로파일(잔차 − 매끄러운 적합)의 fold 간 상관 — tiedprobe 가 밝힌
       신기루 모드("양 fold 지속"은 구조의 존재 재현이지 같은 구조가 아니다) 감별
    C  원시 관계의 시즌 간 지속 (mixdiag 방식: 십분위 모양의 인접 시즌 상관
       + split-half 천장) — r3_ent 는 여기서 −0.21 이었다 (천장 0.962)

주의: _sm 은 근사 재구성이다 (prior = recency 가중 리그 평균, winsorize 생략).
프로파일 진단 목적에는 충분하고, 배포 피처와의 정확 일치를 주장하지 않는다.

사전 등록: 셋 다 통과할 때만 빈 δ 표 β 스윕(별도 실험)으로 승격.
사전 예측: A 에서 z 소멸 또는 B 에서 fold 간 프로파일 상관 낮음.

실행:  .venv/Scripts/python.exe -u src/mixgate.py 2>&1 | tee experiments/mixgate.log
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
from config import RECENCY_HALF_LIFE, SEED, SMOOTH_K_PITCHER
from data import recency_weight
from stepprobe import excess_z

NBINS = 24
NBIN_SHAPE = 10
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]


def rank_bins(v, nbins=NBINS):
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return np.minimum((r * nbins) // len(v), nbins - 1)


def excess_profile(v, r, var_row, nbins=NBINS):
    """등질량 빈의 (평균 잔차 − 3차 다항 적합) 프로파일. fold 간 상관용."""
    b = rank_bins(v, nbins)
    n = np.bincount(b, minlength=nbins).astype(float)
    rbar = np.bincount(b, weights=r, minlength=nbins) / n
    x = (np.arange(nbins) + 0.5) / nbins
    fit = np.polyval(np.polyfit(x, rbar, 3, w=np.sqrt(n)), x)
    return rbar - fit, rbar


def shape10(y, v):
    b = rank_bins(v, NBIN_SHAPE)
    m = np.array([y[b == i].mean() for i in range(NBIN_SHAPE)])
    return m - m.mean()


def main():
    rng = np.random.default_rng(SEED)
    df, s, y = D.load_df()

    cols = {}
    raw_b = df["asof_pitcher_breaking_rate"].to_numpy(np.float64)
    raw_f = df["asof_pitcher_fastball_rate"].to_numpy(np.float64)
    n_mix = np.nan_to_num(df["asof_pitcher_pitchmix_n"].to_numpy(np.float64), nan=0.0)
    cols["fastball_rate(raw)"] = raw_f
    # _sm 근사 재구성은 fold 별로 (prior 가 학습 창 함수)
    print(f"train {df.shape}   K={SMOOTH_K_PITCHER}\n")

    prof = {}
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        tr = s < fold
        w = recency_weight(s[tr], RECENCY_HALF_LIFE, ref_season=int(s[tr].max()))
        m = np.isfinite(raw_b[tr])
        prior = float(np.average(raw_b[tr][m], weights=w[m]))
        sm = np.where(np.isfinite(raw_b), raw_b, prior)
        sm = (n_mix * sm + SMOOTH_K_PITCHER * prior) / (n_mix + SMOOTH_K_PITCHER)
        vals = {"breaking_rate_sm(재구성)": sm, "fastball_rate(raw)": raw_f}

        var_row = st["p0"] * (1 - st["p0"])
        print("=" * 92)
        print(f"fold {fold}   기준 BSS {st['base']:.1f}   (§7.77 당시 z: breaking_sm "
              f"{'7.5' if fold == 2023 else '7.1'} / fastball {'5.9' if fold == 2023 else '5.7'})")
        print("=" * 92)
        for name, v_all in vals.items():
            v = v_all[st['vi']][st['mm']]
            ok = np.isfinite(v)
            z, nb = excess_z(v, st["r"], var_row)
            vsh = rng.permutation(v)
            z0, _ = excess_z(vsh, st["r"], var_row)
            e, rbar = excess_profile(v[ok], st["r"][ok], var_row[ok])
            prof.setdefault(name, {})[fold] = (e, rbar)
            print(f"  A. {name:<26} excess z {z:>+7.2f} (빈 {nb})   셔플 널 {z0:>+6.2f}")

    print(f"\n{'='*92}\n█ B — 초과 프로파일의 fold 간 상관 (신기루 감별)\n{'='*92}")
    for name, byf in prof.items():
        e3, r3 = byf[2023]
        e4, r4 = byf[2024]
        ce = np.corrcoef(e3, e4)[0, 1]
        cr = np.corrcoef(r3, r4)[0, 1]
        print(f"  {name:<28} corr(초과 프로파일) {ce:+.3f}   corr(원 프로파일) {cr:+.3f}")
        print(f"    f2023 초과×1e3: " + " ".join(f"{x*1e3:+5.1f}" for x in e3[::3]))
        print(f"    f2024 초과×1e3: " + " ".join(f"{x*1e3:+5.1f}" for x in e4[::3]))

    print(f"\n{'='*92}\n█ C — 원시 관계의 시즌 간 지속 (mixdiag 방식)\n{'='*92}")
    for name, v_all in (("breaking_rate(raw)", raw_b), ("fastball_rate(raw)", raw_f)):
        shp, rel = {}, []
        for t in SEASONS:
            mts = (s == t) & np.isfinite(v_all)
            yy, vv = y[mts], v_all[mts]
            shp[t] = shape10(yy, vv)
            h = np.arange(len(yy)) % 2                      # split-half 천장
            r_hh = np.corrcoef(shape10(yy[h == 0], vv[h == 0]),
                               shape10(yy[h == 1], vv[h == 1]))[0, 1]
            rel.append(2 * r_hh / (1 + r_hh))
        adj = [np.corrcoef(shp[t], shp[t + 1])[0, 1] for t in SEASONS[:-1]]
        ceil = float(np.mean(rel))
        print(f"  {name:<24} 인접 시즌 모양 상관: "
              + " ".join(f"{c:+.2f}" for c in adj)
              + f"   평균 {np.mean(adj):+.3f}   천장 {ceil:.3f}"
              f"   비율 {np.mean(adj)/ceil:+.3f}")
    print("\n  참고: 채택 전이 성공 축 gap_success 0.82 / 실패 축 r3_ent −0.21 (§7.59)")
    print("판정: A z 소멸 또는 B corr 낮음 → 신기루로 종결. C 비율 <0.7 → 조건 ④ 미달.")


if __name__ == "__main__":
    main()
