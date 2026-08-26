"""신인(무이력 선수)에 대한 **s37 챔피언 층** 성능 부검 — 리그·연도·실력·시점 축.

§7.79~§7.81 이 같은 질문을 s23 층에서 물었다. 그 뒤 층이 크게 바뀌었다
(init_score · CatBoost12 · fspec · MLP 변형가중 · 확률 아핀). memory:
"닫은 진단도 새 도구로 다시 물어라". 그리고 §7.151(APN_SHIFT)은 이 축에
보정을 걸었다가 s38 에서 롤백됐다 — 무엇이 남아 있는지 다시 센다.

하네스 검증(§0)이 먼저다. s29 층을 재현하지 못하면(807.2 / 924.5) 아래 표는
전부 내 버그를 재는 것이다. memory: "간이 재구현은 알려진 수치 재현부터".

실행:  .venv/Scripts/python.exe -u src/rookdiag.py 2>&1 | tee experiments/rookdiag.log
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

import foldcache as _fc
from config import (CAT_ALPHA, CAT_SEEDS, CAT_VARIANTS, ASOF_AXES, CTX_AXES,
                    FSPEC_ALPHA, MLP_ALPHA, MLP_SEEDS, MLP_VARIANT_WEIGHT,
                    MLP_VARIANTS, ROOT, TARGET_COL, TRAIN_CSV)
from data import load_csv
from predict import PROB_CENTER, PROB_SCALE
from validation import bss, brier, shift_to_prior

CACHE = ROOT / "experiments" / "cache"
KEY = _fc._key()
FOLDS = [2023, 2024]
EXPECT_S29 = {2023: 807.2, 2024: 924.5}      # clipdesign.log 실측 — 하네스 검증용
B = 1000
RNG = np.random.default_rng(42)


# ── 층 조립 ──────────────────────────────────────────────────────
def build(fold, df, s):
    """s37 배포 층 그대로 fold 예측을 만든다. 중간 층 BSS 도 같이 돌려준다."""
    va = df[s == fold]
    y = va[TARGET_COL].to_numpy(dtype=np.float64)
    ybar = y.mean()
    gtF = va["game_type"].to_numpy().astype(str) == "F"

    with np.load(CACHE / f"dep29_fold{fold}_{KEY}.npz", allow_pickle=True) as z:
        Xax = z["Xax"]
    with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d0:
        assert np.allclose(d0["y"], y), "캐시 y 불일치 — 지문이 틀렸다"
        z_mlp_flat, mem = d0["z_mlp"], d0["members"]
    z_lgb = np.load(CACHE / f"initdep_fold{fold}_ctr0.npz")["z"]
    ax = Xax @ np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    # 변형 가중 MLP — foldcache 의 z_mlp 는 평평 평균이라 배포와 다르다
    i, zs = 0, {}
    for vname, (_, nseed) in MLP_VARIANTS.items():
        zs[vname] = np.mean(mem[i:i + nseed], axis=0)
        i += nseed
    assert i == len(mem), (i, len(mem))
    z_mlp = sum(MLP_VARIANT_WEIGHT[v] * z for v, z in zs.items())

    z_cat = np.mean([np.load(CACHE / f"catcpu_fold{fold}_{v}_{sd}.npz")["z"]
                     for v in CAT_VARIANTS for sd in CAT_SEEDS], axis=0)

    def score(z):
        return bss(y, shift_to_prior(1 / (1 + np.exp(-(z + ax))), ybar))

    layers = {}
    layers["s29 (평평 MLP)"] = score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp_flat)
    z = (1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * z_mlp
    layers["+ 변형가중 MLP"] = score(z)
    fs = CACHE / f"s35_fspec_fold{fold}.npz"
    if fs.exists():
        zf = np.load(fs)["z"]
        z = z.copy()
        z[gtF] = (1 - FSPEC_ALPHA) * z[gtF] + FSPEC_ALPHA * zf[gtF]
        layers["+ fspec(F)"] = score(z)
    else:
        layers["+ fspec(F)"] = float("nan")
    z = (1 - CAT_ALPHA) * z + CAT_ALPHA * z_cat
    layers["+ CatBoost12"] = score(z)

    p = shift_to_prior(1 / (1 + np.exp(-(z + ax))), ybar)
    p = shift_to_prior(PROB_CENTER + PROB_SCALE * (p - PROB_CENTER), ybar)
    layers["+ 확률 아핀 (= s37)"] = bss(y, p)
    return va, y, p, layers


# ── 코호트 ───────────────────────────────────────────────────────
def prior_seasons(df, s, fold, col):
    hist = df.loc[s < fold, [col, "season"]]
    cnt = hist.groupby(col, observed=True)["season"].nunique()
    return cnt.reindex(df.loc[s == fold, col].to_numpy()).fillna(0).to_numpy().astype(int)


def cells(y, p, m, ref_all, bss_all):
    yg, pg = y[m], p[m]
    w = m.mean()
    ref_g = yg.mean() * (1 - yg.mean())
    loc = 1e5 * (1 - brier(yg, pg) / ref_g)
    return dict(n=int(m.sum()), w=w, ybar=yg.mean(), pbar=pg.mean(),
                bias=pg.mean() - yg.mean(), loc=loc,
                contrib=w * 1e5 * (1 - brier(yg, pg) / ref_all),
                head=w * (ref_g / ref_all) * (bss_all - loc))


def table(title, rows, y, p, ref_all, bss_all):
    print(f"\n  {title}")
    print(f"    {'그룹':<24}{'n':>10}{'비중':>7}{'ȳ':>8}{'p̄':>8}"
          f"{'편향':>9}{'로컬BSS':>10}{'기여':>9}{'잠재이득':>10}")
    for name, m in rows:
        if m.sum() < 50:
            print(f"    {name:<24}{'—':>10}")
            continue
        c = cells(y, p, m, ref_all, bss_all)
        print(f"    {name:<24}{c['n']:>10,}{c['w']*100:>6.1f}%{c['ybar']:>8.4f}"
              f"{c['pbar']:>8.4f}{c['bias']:>+9.4f}{c['loc']:>10.1f}"
              f"{c['contrib']:>9.1f}{c['head']:>+10.1f}")


def cl_boot(cl, n_id, y, p, m1, m2, stat="bias"):
    """투수 클러스터 부트스트랩으로 (그룹1 − 그룹2) 대비의 구간을 낸다."""
    def acc(mask, w):
        return (np.bincount(cl, weights=mask * w, minlength=n_id))
    n1, n2 = acc(m1, np.ones_like(y)), acc(m2, np.ones_like(y))
    if stat == "bias":
        s1, s2 = acc(m1, p - y), acc(m2, p - y)
    else:                                    # 로컬 BSS 는 그룹 ref 가 필요하다
        s1, s2 = acc(m1, (p - y) ** 2), acc(m2, (p - y) ** 2)
        y1, y2 = acc(m1, y), acc(m2, y)
    out = np.empty(B)
    idx = np.arange(n_id)
    for b in range(B):
        k = RNG.integers(0, n_id, n_id)
        a1, a2 = n1[k].sum(), n2[k].sum()
        if a1 == 0 or a2 == 0:
            out[b] = np.nan
            continue
        if stat == "bias":
            out[b] = s1[k].sum() / a1 - s2[k].sum() / a2
        else:
            m_1, m_2 = y1[k].sum() / a1, y2[k].sum() / a2
            r1, r2 = m_1 * (1 - m_1), m_2 * (1 - m_2)
            out[b] = (1e5 * (1 - s1[k].sum() / a1 / r1)
                      - 1e5 * (1 - s2[k].sum() / a2 / r2))
    o = out[~np.isnan(out)]
    return o.mean(), np.percentile(o, [2.5, 97.5]), o.mean() / o.std()


def wslope(g, y, p, n_id):
    """투수 단위 캘리브레이션 기울기 — ȳ_j 를 p̄_j 에 회귀 (가중 n_j).
    <1 = 투수 간 차이를 과대 예측, >1 = 과소(압축)."""
    n = np.bincount(g, minlength=n_id)
    sy = np.bincount(g, weights=y, minlength=n_id)
    sp = np.bincount(g, weights=p, minlength=n_id)
    m = n > 0
    n, yb, pb = n[m], sy[m] / np.maximum(n[m], 1), sp[m] / np.maximum(n[m], 1)
    P, Y = np.average(pb, weights=n), np.average(yb, weights=n)
    num = np.sum(n * (pb - P) * (yb - Y))
    den = np.sum(n * (pb - P) ** 2)
    out = np.empty(B)
    K = len(n)
    for b in range(B):
        k = RNG.integers(0, K, K)
        nn, pp, yy = n[k], pb[k], yb[k]
        Pb, Yb = np.average(pp, weights=nn), np.average(yy, weights=nn)
        d = np.sum(nn * (pp - Pb) ** 2)
        out[b] = np.sum(nn * (pp - Pb) * (yy - Yb)) / d if d > 0 else np.nan
    o = out[~np.isnan(out)]
    return (num / den, np.percentile(o, [2.5, 97.5]), int(m.sum()),
            np.sqrt(np.average((pb - P) ** 2, weights=n)))


def within_slope(g, y, p, n_id):
    """투수 **내부** 기울기 (§7.80 과 같은 정의). 그룹 평균 제거 후 원점 회귀."""
    n = np.bincount(g, minlength=n_id)
    mp = np.bincount(g, weights=p, minlength=n_id) / np.maximum(n, 1)
    my = np.bincount(g, weights=y, minlength=n_id) / np.maximum(n, 1)
    dp, dy = p - mp[g], y - my[g]
    num = np.bincount(g, weights=dp * dy, minlength=n_id)
    den = np.bincount(g, weights=dp * dp, minlength=n_id)
    out = np.empty(B)
    seen = np.where(n > 0)[0]
    for b in range(B):
        k = RNG.choice(seen, len(seen), replace=True)
        d = den[k].sum()
        out[b] = num[k].sum() / d if d > 0 else np.nan
    o = out[~np.isnan(out)]
    return num.sum() / den.sum(), np.percentile(o, [2.5, 97.5])


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()

    print("=" * 104)
    print(f"§0  하네스 검증 — 층별 BSS (지문 {KEY})")
    print("=" * 104)
    ST = {}
    for fold in FOLDS:
        va, y, p, layers = build(fold, df, s)
        ST[fold] = (va, y, p)
        print(f"\n  fold {fold}  n={len(y):,}  ȳ={y.mean():.4f}")
        for k, v in layers.items():
            mark = ""
            if k.startswith("s29"):
                d = v - EXPECT_S29[fold]
                mark = (f"   기대 {EXPECT_S29[fold]}  Δ{d:+.2f}  "
                        + ("OK" if abs(d) < 0.3 else "!! 어긋남"))
            print(f"    {k:<24}{v:>10.1f}{mark}")

    print("\n" + "=" * 104)
    print("§1  코호트 규모 — 신인은 몇 퍼센트인가, 그리고 늘고 있는가")
    print("=" * 104)
    hdr = ("검증시즌", "행", "투수신인행", "투수신인수", "타자신인행", "타자신인수", "F행")
    print(f"\n    {hdr[0]:<10}{hdr[1]:>12}{hdr[2]:>12}{hdr[3]:>11}"
          f"{hdr[4]:>12}{hdr[5]:>11}{hdr[6]:>8}")
    for fold in [2021, 2022, 2023, 2024]:
        m = s == fold
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pb = prior_seasons(df, s, fold, "batter_id")
        npr = df.loc[m, "pitcher_id"].to_numpy()[pp == 0]
        nbr = df.loc[m, "batter_id"].to_numpy()[pb == 0]
        gf = (df.loc[m, "game_type"].to_numpy().astype(str) == "F").mean()
        print(f"    {fold:<10}{m.sum():>12,}{(pp==0).mean()*100:>11.1f}%"
              f"{len(np.unique(npr)):>11,}{(pb==0).mean()*100:>11.1f}%"
              f"{len(np.unique(nbr)):>11,}{gf*100:>7.1f}%")

    SUM = {}
    for fold in FOLDS:
        va, y, p = ST[fold]
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pb = prior_seasons(df, s, fold, "batter_id")
        isF = va["game_type"].to_numpy().astype(str) == "F"
        an = va["asof_pitcher_n"].to_numpy(dtype=np.float64)
        mon = va["game_month"].to_numpy()
        g, uniq = pd.factorize(va["pitcher_id"].to_numpy())
        n_id = len(uniq)
        ref_all = y.mean() * (1 - y.mean())
        bss_all = bss(y, p)
        rook, vet = pp == 0, pp >= 2

        print("\n" + "=" * 104)
        print(f"§2  fold {fold}  (전체 BSS {bss_all:.1f}) — 경력 x 리그")
        print("=" * 104)
        table("경력(직전 시즌들에 등장한 시즌 수)",
              [("0시즌 = 신인", pp == 0), ("1시즌", pp == 1),
               ("2시즌", pp == 2), ("3+시즌", pp >= 3)], y, p, ref_all, bss_all)
        for lab, gm in (("R (정규)", ~isF), ("F (그 외)", isF)):
            table(f"{lab} 안에서",
                  [("  신인", (pp == 0) & gm), ("  1시즌", (pp == 1) & gm),
                   ("  2+시즌", (pp >= 2) & gm)], y, p, ref_all, bss_all)

        print("\n" + "=" * 104)
        print(f"§3  fold {fold} — 캘리브레이션 기울기 (투수 간 / 투수 내)")
        print("=" * 104)
        h = ("코호트", "투수수", "sd(p_j)", "투수간기울기", "95% CI", "투수내기울기", "95% CI")
        print(f"    {h[0]:<16}{h[1]:>8}{h[2]:>10}{h[3]:>16}{h[4]:>22}{h[5]:>16}{h[6]:>22}")
        for lab, m in (("신인", rook), ("1시즌", pp == 1), ("2+시즌", vet)):
            gg = g[m]
            b1, ci1, k, sd = wslope(gg, y[m], p[m], n_id)
            b2, ci2 = within_slope(gg, y[m], p[m], n_id)
            print(f"    {lab:<16}{k:>8,}{sd:>10.4f}{b1:>16.3f}"
                  f"{f'[{ci1[0]:+.3f}, {ci1[1]:+.3f}]':>22}{b2:>16.3f}"
                  f"{f'[{ci2[0]:+.3f}, {ci2[1]:+.3f}]':>22}")

        print("\n" + "=" * 104)
        print(f"§4  fold {fold} — 실력별 (반쪽A로 실력 추정 -> 반쪽B에서 평가)")
        print("=" * 104)
        half = RNG.random(len(y)) < 0.5
        nA = np.bincount(g, weights=half, minlength=n_id)
        sA = np.bincount(g, weights=half * y, minlength=n_id)
        rateA = np.where(nA > 0, sA / np.maximum(nA, 1), np.nan)
        okA = nA >= 100
        for lab, m in (("신인", rook), ("2+시즌", vet)):
            sel = m & (~half) & okA[g]
            if sel.sum() < 200:
                print(f"\n    {lab}: 표본 부족")
                continue
            r = rateA[g]
            qs = np.nanquantile(r[sel], [.2, .4, .6, .8])
            bkt = np.digitize(r, qs)
            rows = [(f"  {lab} 실력 Q{i+1}", sel & (bkt == i)) for i in range(5)]
            table(f"{lab} — 반쪽A 성공률 5분위 (평가는 반쪽B)", rows, y, p, ref_all, bss_all)

        print("\n" + "=" * 104)
        print(f"§5  fold {fold} — asof_pitcher_n (누적 표본 수) 버킷")
        print("=" * 104)
        edges = [1, 100, 500, 2000, 10000]
        lab_n = ["결측", "1~99", "100~499", "500~1999", "2000~9999", "10000+"]
        bn = np.where(np.isnan(an), 0, np.digitize(an, edges) + 1)
        table("전체", [(lab_n[i], bn == i) for i in range(6)], y, p, ref_all, bss_all)
        table("신인만", [("  신인 " + lab_n[i], (bn == i) & rook) for i in range(6)],
              y, p, ref_all, bss_all)

        print("\n" + "=" * 104)
        print(f"§6  fold {fold} — 월별 궤적 (신인 적응)")
        print("=" * 104)
        h = ("월", "신인 n", "신인 y", "신인 편향", "신인 로컬", "2+ n", "2+ y", "2+ 편향", "2+ 로컬")
        print(f"    {h[0]:<6}{h[1]:>10}{h[2]:>9}{h[3]:>11}{h[4]:>11}"
              f"{h[5]:>11}{h[6]:>9}{h[7]:>10}{h[8]:>10}")
        for mo in sorted(np.unique(mon)):
            a, b = rook & (mon == mo), vet & (mon == mo)
            if a.sum() < 200 or b.sum() < 200:
                continue
            ca = cells(y, p, a, ref_all, bss_all)
            cb = cells(y, p, b, ref_all, bss_all)
            print(f"    {mo:<6}{ca['n']:>10,}{ca['ybar']:>9.4f}{ca['bias']:>+11.4f}"
                  f"{ca['loc']:>11.1f}{cb['n']:>11,}{cb['ybar']:>9.4f}"
                  f"{cb['bias']:>+10.4f}{cb['loc']:>10.1f}")

        print("\n" + "=" * 104)
        print(f"§7  fold {fold} — 타자 신인 · 투타 교차")
        print("=" * 104)
        table("타자 경력", [("타자 신인", pb == 0), ("타자 1시즌", pb == 1),
                        ("타자 2+시즌", pb >= 2)], y, p, ref_all, bss_all)
        table("투수 x 타자 신인 교차",
              [("투신 x 타신", rook & (pb == 0)), ("투신 x 타2+", rook & (pb >= 2)),
               ("투2+ x 타신", vet & (pb == 0)), ("투2+ x 타2+", vet & (pb >= 2))],
              y, p, ref_all, bss_all)

        SUM[fold] = dict(y=y, p=p, cl=g, n_id=n_id, rook=rook, vet=vet, isF=isF)

    print("\n" + "=" * 104)
    print("§8  신인 − 2+시즌 대비 (투수 클러스터 부트스트랩 B=1000)")
    print("=" * 104)
    h = ("대비", "fold", "delta", "95% CI", "t")
    print(f"\n    {h[0]:<28}{h[1]:>6}{h[2]:>12}{h[3]:>26}{h[4]:>7}")
    for fold in FOLDS:
        d = SUM[fold]
        for lab, m1, m2 in (
            ("편향 (신인−2+)", d["rook"], d["vet"]),
            ("편향 R 안에서", d["rook"] & ~d["isF"], d["vet"] & ~d["isF"]),
            ("편향 F 안에서", d["rook"] & d["isF"], d["vet"] & d["isF"]),
        ):
            mu, ci, t = cl_boot(d["cl"], d["n_id"], d["y"], d["p"], m1, m2, "bias")
            print(f"    {lab:<28}{fold:>6}{mu:>+12.4f}"
                  f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]':>26}{t:>+7.1f}")
        for lab, m1, m2 in (
            ("로컬BSS (신인−2+)", d["rook"], d["vet"]),
            ("로컬BSS R 안에서", d["rook"] & ~d["isF"], d["vet"] & ~d["isF"]),
            ("로컬BSS F 안에서", d["rook"] & d["isF"], d["vet"] & d["isF"]),
        ):
            mu, ci, t = cl_boot(d["cl"], d["n_id"], d["y"], d["p"], m1, m2, "bss")
            print(f"    {lab:<28}{fold:>6}{mu:>+12.1f}"
                  f"{f'[{ci[0]:+.1f}, {ci[1]:+.1f}]':>26}{t:>+7.1f}")


if __name__ == "__main__":
    main()
