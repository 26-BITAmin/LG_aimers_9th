"""§7.205 — `soft2`(투수별 세분 soft label) 재측정. 3시드 -> **10시드**.

왜 다시 재는가 (§7.70 1단계는 기각됐다)

    §7.70   Δ(soft2−soft)  f2024 **+4.0±1.4**  f2023 **−3.6±2.3**  -> 조건 ②(부호 일치) 위반으로 기각
    §7.200  "± 가 행 부트스트랩이라 draw 잡음 미포함 — 재측정 대상" 이라 적었다.

★ 그 진단이 사실관계로 틀렸다. `softlabel2.py:127` 은

        d = res[("soft2",fold)][:,0] - res[("soft",fold)][:,0]      # 같은 시드끼리 쌍체
        print(f"Δ = {d.mean():+.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):.1f}")

    **시드 3개에 대한 쌍체 SE** 다 — 순수 draw 잡음이고 행 부트스트랩 성분이 없다.
    (그리고 `train_eval` 은 `lgb.train` 만 돈다 — 감사표의 "LGBM+MLP" 층 표기도 틀렸다.)

★★ 진짜 문제는 **df = 2** 였다.

        t(.975, df 2) = 4.303   (1.96 이 아니다)
        f2024  +4.0 ± 1.4  ->  95% CI [ −2.02, +10.02]   0 포함
        f2023  −3.6 ± 2.3  ->  95% CI [−13.50,  +6.30]   0 포함

    **두 fold 다 0 을 포함한다 = 부호 분열이 아니다.** §7.111 이 개정한 규칙
    ("어느 fold 도 0 을 유의하게 밑돌지 않을 것")을 통과하는 자리이고,
    그 자리에서 채택된 `init_score` 는 LB +5.30 신기록이었다 (§7.200 D군).

★ 같은 일이 챔피언에게도 있었다 — `ablate17` 의 `emb8` 팔

        ② 아키텍처   emb4 +21.2(1.6) / emb8 +17.0(2.5) / noid +27.0(3.1 ✗)
        ① 시드 수     x1 +17.8 / x2 +19.1 / **x3 +18.3** / **x5 +21.2**

    `emb8` 은 게이트를 통과했는데(크기비 2.5 ≤ 3) emb4 보다 4.2 낮아 안 뽑혔다.
    그런데 `emb8` 은 **3시드**, `emb4` 는 **5시드**였고 같은 표 ①이 3→5 를 **+2.9** 로
    재놨다. 예산 보정 후 격차는 **−1.3** — 잡음 이하다. 축을 격리하니 **LB +7.67**,
    지금 챔피언이다. ([[preregister-power-not-just-thresholds]])

    -> 두 사건 다 `n=3` 이 결론을 만들었다. 한 번은 df 를 키워 판정 불능으로,
       한 번은 점추정을 끌어내려 순위 역전으로.

이 스크립트

    시드 10개 · 같은 시드끼리 쌍체(CRN) · `soft` 와 `soft2` 두 팔만 (base 생략).
    §7.70 과 **같은 하네스·같은 h 산식**을 쓴다 — `softlabel2` 에서 그대로 import 한다.
    df 9 -> t(.975,9) = 2.262. 관측 SD 를 그대로 쓰면 f2024 SE 1.4 -> 0.77.

⚠️ 사전 등록 — 이 실행 전에 정한다
    판정은 **§7.111 개정 규칙**: "어느 fold 도 0 을 유의하게 밑돌지 않을 것"
    (95% CI 하한 > 0 을 요구하지 않는다. **상한 < 0 이면** 기각).
    부호 일치는 요구하지 않는다 — §7.200 D군이 그 규칙으로 3승 1패다.
    크기 전망은 soft label 본체의 실측 전이 배율 0.175 (f2024 +17.7 -> LB +3.09).

실행:  .venv/Scripts/python.exe -u src/softlabel3.py > experiments/softlabel3.log 2>&1
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

from config import TARGET_COL, TRAIN_CSV
from data import load_csv
from softlabel import fit_h
from softlabel2 import make_h_rows, prep, rev_propensity, run_soft

FOLDS = [2024, 2023]
SEEDS = (42, 7, 2024, 1234, 31337, 99, 2718, 314, 1618, 577)
TCRIT = {9: 2.262157, 2: 4.302653}          # t(.975, df)
XFER = 3.09 / 17.7                          # soft label 본체 실측 전이 배율


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    print(f"§7.205  soft2 재측정 — 시드 {len(SEEDS)}개 (쌍체 CRN), df {len(SEEDS)-1}")
    print(f"  시드 {SEEDS}")
    print(f"  §7.70 원본: f2024 +4.0±1.4 · f2023 −3.6±2.3  (3시드, df 2)\n")

    out = {}
    for fold in FOLDS:
        print(f"=== fold {fold} (train < {fold}) ===")
        tr, va, w, X, art = prep(df, fold)
        ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
        h = fit_h(df, (df.season < fold).to_numpy())
        r_sm = rev_propensity(tr, art)

        arms = {}
        for arm, rs in [("soft", None), ("soft2", r_sm)]:
            h_row = make_h_rows(tr, ytr, h, rs)
            arms[arm] = run_soft(tr, va, w, X, h_row, SEEDS)[:, 0]
            print(f"  {arm:<6} BSS {arms[arm].mean():>8.2f} ± {arms[arm].std(ddof=1):.2f}"
                  f"   ({time.time()-t0:.0f}s)")

        d = arms["soft2"] - arms["soft"]          # 쌍체 (같은 시드 인덱스)
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = d.mean() - TCRIT[n - 1] * se, d.mean() + TCRIT[n - 1] * se
        out[fold] = (d.mean(), se, lo, hi, d)
        print(f"  → Δ(soft2−soft) = {d.mean():+.2f} ± {se:.2f}   t {d.mean()/se:+.2f}"
              f"   95% CI [{lo:+.2f}, {hi:+.2f}]   양수 {int((d>0).sum())}/{n}")
        print(f"     시드별 Δ: {np.array2string(d, precision=1, floatmode='fixed')}\n")
        del tr, va, w, X

    print("=" * 78)
    print("판정 — §7.111 개정 규칙: 어느 fold 도 0 을 유의하게 밑돌지 않을 것")
    print("=" * 78)
    print(f"  {'fold':<8}{'Δ':>9}{'SE':>7}{'t':>8}{'95% CI':>24}{'상한<0?':>10}")
    bad = []
    for f in FOLDS:
        m, se, lo, hi, _ = out[f]
        below = hi < 0
        bad += [f] if below else []
        print(f"  {f:<8}{m:>+9.2f}{se:>7.2f}{m/se:>+8.2f}   [{lo:+8.2f}, {hi:+8.2f}]"
              f"{'예 (기각)' if below else '아니오':>10}")
    print()
    if bad:
        print(f"  -> fold {bad} 에서 0 을 유의하게 밑돈다. **기각.**")
    else:
        print("  -> 개정 규칙 통과. §7.70 의 조건 ② 기각은 근거가 없었다.")
        m24 = out[2024][0]
        print(f"     크기 전망: f2024 {m24:+.2f} × 전이 0.175  ->  LB {m24*XFER:+.2f}")
    print(f"\n  ※ 3시드(df 2) 대비 변화: f2024 +4.0±1.4 -> {out[2024][0]:+.2f}±{out[2024][1]:.2f}"
          f" · f2023 −3.6±2.3 -> {out[2023][0]:+.2f}±{out[2023][1]:.2f}")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
