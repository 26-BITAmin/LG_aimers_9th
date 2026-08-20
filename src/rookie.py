"""신인(직전 시즌 이력 없음) × game_type(R/F) 성능 분해 — 배포 구성 s23 기준.

질문 (사용자)
    "성능 문제가 어디서 나오나. R·F 각각에서 새로 막 들어온(한 시즌만 뛴)
     선수들을 얼마나 잘 맞췄나?"

무엇을 재나
    foldcache(현행 지문)의 z_lgb·z_mlp에 배포와 같은 사후 보정을 얹어
    (§PIPELINE.md §0) fold 예측 P를 만든다. 그 다음 행을 쪼갠다:
        · game_type R / F
        · 투수 경력 = 검증 시즌 **이전** 시즌 중 등장한 시즌 수 (0 = 신인)
        · 타자 경력도 같은 방식
    지표 두 개를 같이 낸다 — 섞으면 오독한다.
        로컬 BSS  = 1e5·(1 − BS_g / ȳ_g(1−ȳ_g))   그룹 안에서 뽑은 신호
        총점 기여 = w_g · 1e5·(1 − BS_g / ref_all) 합하면 전체 BSS가 된다
        잠재이득  = w_g · (ref_g/ref_all) · (BSS_all − BSS_g)
                    이 그룹의 스킬이 fold 평균까지만 올라왔을 때의 총점 증가.
                    ⚠️ 단순히 (BS_g − BS_all)을 쓰면 **기저 분산 차이**가 섞인다 —
                       ȳ가 .5에 가까운 그룹은 잘 맞춰도 Brier가 크다. ref_g로 나눠 지운다.

주의
    prior shift는 fold 전체 평균으로 맞추는 **오라클**이다 (저장소 관례).
    따라서 그룹별 편향(p̄ − ȳ)은 0으로 강제되지 않고 그대로 드러난다.

실행:  .venv/Scripts/python.exe -u src/rookie.py 2>&1 | tee experiments/rookie.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import asofdelta
import ctxfit
import features as F
import foldcache
# ⚠️ **지문을 못 박는다.** 이 분석의 대상은 배포 중인 s23 구성이다.
#    다른 세션이 config를 바꾸면 foldcache 지문이 따라 바뀌고, load()가
#    "캐시 없음"으로 판단해 fold당 20분짜리 재학습을 시작한다 (실제로 발생 —
#    달력 피처 DROP 후보가 들어와 faa19e0e95 → 5488ef1b18).
#    새 구성으로 다시 재려면 이 상수를 지우거나 새 지문으로 바꾼다.
#    기본값은 s23(§7.79~7.81 기록의 기준). s24 이후 구성으로 재측정할 땐
#    환경변수로 지문을 넘긴다:  LGA_FOLD_KEY=5488ef1b18
#    "auto"를 주면 현재 config에서 계산한 지문을 쓴다 (핀 해제).
import os as _os
_PIN = _os.environ.get("LGA_FOLD_KEY") or "faa19e0e95"
if _PIN != "auto":
    _orig_key = foldcache._key
    foldcache._key = lambda: _PIN
FOLD_KEY = _PIN

from config import (ASOF_AXES, ASOF_STRENGTH, CONTAM_AXES, CTX_AXES, CTX_HALF_LIFE,
                    CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from validation import bss, brier, shift_to_prior

FOLDS = [2023, 2024]


def deployed_pred(df, s, fold, cached):
    """PIPELINE.md §0 그대로: z = .6·lgb + .4·mlp → +δ_ctx +δ_asof → prior shift."""
    tr_m, va_m = s < fold, s == fold
    tr, va = df[tr_m], df[va_m].copy()
    yv = cached[fold][2]
    z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

    ref = int(s[tr_m].max())
    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K,
                         weights=recency_weight(s[tr_m], CTX_HALF_LIFE, ref_season=ref))
    art = F.fit_artifacts(tr, ctx=ctx, contam_axes=CONTAM_AXES)
    art["asof"] = asofdelta.fit_asof_axes(tr, ASOF_AXES, ASOF_STRENGTH,
                                          ct=art.get("contam"))
    adj, _ = F.logit_adjust(va, art)
    p = shift_to_prior(1.0 / (1.0 + np.exp(-(z + adj))), yv.mean())
    assert np.array_equal(yv, va[TARGET_COL].to_numpy(dtype=np.float64))
    return va, yv, p


def prior_seasons(df, s, fold, col):
    """검증 시즌 이전에 그 선수가 등장한 **시즌 수**를 행 단위로 붙인다."""
    hist = df.loc[s < fold, [col, "season"]]
    cnt = hist.groupby(col, observed=True)["season"].nunique()
    ids = df.loc[s == fold, col].to_numpy()
    return cnt.reindex(ids).fillna(0).to_numpy().astype(int)


def total_seasons(df, col):
    """2019~2024 전체에서 그 선수가 뛴 시즌 수 (= '한 시즌만 뛴' 판정용)."""
    return df.groupby(col, observed=True)["season"].nunique()


def table(title, rows, y, p, ref_all, bss_all, _unused):
    print(f"\n  {title}")
    print(f"    {'그룹':<22}{'n':>10}{'비중':>7}{'ȳ':>8}{'p̄':>8}"
          f"{'편향':>9}{'로컬BSS':>10}{'기여':>9}{'잠재이득':>10}")
    for name, m in rows:
        if m.sum() == 0:
            print(f"    {name:<22}{'—':>10}")
            continue
        yg, pg = y[m], p[m]
        w = m.mean()
        bs = brier(yg, pg)
        loc = bss(yg, pg) if 0 < yg.mean() < 1 else float("nan")
        ref_g = yg.mean() * (1 - yg.mean())
        contrib = w * 1e5 * (1 - bs / ref_all)
        head = w * (ref_g / ref_all) * (bss_all - loc)
        print(f"    {name:<22}{m.sum():>10,}{w*100:>6.1f}%{yg.mean():>8.4f}"
              f"{pg.mean():>8.4f}{pg.mean()-yg.mean():>+9.4f}"
              f"{loc:>10.1f}{contrib:>9.1f}{head:>+10.1f}")


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    cached = foldcache.load(FOLDS, verbose=False)
    tot_p = total_seasons(df, "pitcher_id")
    tot_b = total_seasons(df, "batter_id")

    for fold in FOLDS:
        va, y, p = deployed_pred(df, s, fold, cached)
        gt_f = va["game_type"].astype("object").to_numpy() == "F"
        pp = prior_seasons(df, s, fold, "pitcher_id")
        pb = prior_seasons(df, s, fold, "batter_id")
        one_p = tot_p.reindex(va["pitcher_id"].to_numpy()).to_numpy() == 1
        one_b = tot_b.reindex(va["batter_id"].to_numpy()).to_numpy() == 1
        an = va["asof_pitcher_n"].to_numpy()

        ref_all = y.mean() * (1 - y.mean())
        bss_all = bss(y, p)
        print("\n" + "=" * 96)
        print(f"fold {fold}  n={len(y):,}  ȳ={y.mean():.4f}  BSS={bss_all:.1f}"
              f"   (F행 {gt_f.mean()*100:.1f}%,  투수 신인행 {(pp==0).mean()*100:.1f}%,"
              f"  타자 신인행 {(pb==0).mean()*100:.1f}%)")

        for gname, gm in (("R", ~gt_f), ("F", gt_f)):
            rows = [(f"{gname} 전체", gm)]
            for lo, hi, lab in ((0, 0, "투수 신인(0시즌)"), (1, 1, "투수 1시즌"),
                                (2, 99, "투수 2+시즌")):
                rows.append((f"{gname} {lab}", gm & (pp >= lo) & (pp <= hi)))
            for lo, hi, lab in ((0, 0, "타자 신인(0시즌)"), (1, 1, "타자 1시즌"),
                                (2, 99, "타자 2+시즌")):
                rows.append((f"{gname} {lab}", gm & (pb >= lo) & (pb <= hi)))
            rows.append((f"{gname} 둘 다 신인", gm & (pp == 0) & (pb == 0)))
            rows.append((f"{gname} 둘 다 베테랑", gm & (pp > 0) & (pb > 0)))
            table(f"[{gname}] game_type", rows, y, p, ref_all, bss_all, len(y))

        # 신인 안에서 — 시즌 중 누적 표본(asof_n)이 성능을 가르는가
        rook = pp == 0
        rows = []
        for gname, gm in (("R", ~gt_f), ("F", gt_f)):
            for lo, hi, lab in ((0, 0, "n=0"), (1, 200, "n=1~200"),
                                (201, 1000, "n=201~1k"), (1001, 10**9, "n>1k")):
                rows.append((f"{gname} 신인 {lab}", gm & rook & (an >= lo) & (an <= hi)))
        table("[투수 신인] 시즌 중 누적 투구수(asof_pitcher_n)별",
              rows, y, p, ref_all, bss_all, len(y))

        # '한 시즌만 뛴' = 2019~2024 통틀어 1시즌 (단기 콜업 포함)
        rows = []
        for gname, gm in (("R", ~gt_f), ("F", gt_f)):
            rows.append((f"{gname} 투수 통산1시즌", gm & one_p))
            rows.append((f"{gname} 타자 통산1시즌", gm & one_b))
            rows.append((f"{gname} 둘 다 통산1시즌", gm & one_p & one_b))
        table("['한 시즌만 뛴' — 2019~2024 통산 1시즌]",
              rows, y, p, ref_all, bss_all, len(y))


if __name__ == "__main__":
    main()
