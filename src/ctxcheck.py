"""리팩터 검증 — 배포 경로를 건드렸으니 세 가지를 못 박고 넘어간다

§7.36에서 플래툰을 넣을 때 같은 로직이 네 파일에 흩어져 있어 `verify_pipeline.py`를
빠뜨렸고, **검증이 플래툰 없는 모델을 통과시켰다.** 이번엔 그 로직을
`features.logit_adjust` 하나로 합쳤다. 합치는 과정에서 뭔가 달라지지 않았는지 확인한다.

  ① **비트 동일성** — `ctxfit.fit_ctx`의 hand 축이 옛 `platoonfit.fit_platoon`과
     완전히 같은 배열을 내는가. 다르면 s11에서 잰 +5.93이 이 코드의 것이 아니다.

  ② **측정 재현** — `logit_adjust`로 만든 로짓이 `ctxsplit2.py`의 측정과 같은
     BSS를 주는가. 배포 경로와 측정 경로가 갈라지면 숫자가 의미를 잃는다.

  ③ **행 독립성** — 조각으로 나눠 부른 것과 통짜가 같은가.
     대회 규정상 각 행은 완전히 독립적으로 예측돼야 한다 (CLAUDE.md §3.2).

실행:  .venv/Scripts/python.exe -u src/ctxcheck.py 2>&1 | tee experiments/ctxcheck.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import ctxfit
import foldcache
from config import CTX_AXES, CTX_K, MLP_ALPHA, TARGET_COL, TRAIN_CSV
from data import load_csv
from features import logit_adjust
from validation import bss, shift_to_prior

FOLD = 2024
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    tr = df[s < FOLD]
    va = df[s == FOLD]
    print(f"train {len(tr):,} → valid({FOLD}) {len(va):,}\n")

    ctx = ctxfit.fit_ctx(tr, TARGET_COL, CTX_AXES, CTX_K)
    for line in ctxfit.summary(ctx):
        print(f"  {line}")
    print()

    # ── ① 비트 동일성: hand 축 vs 옛 platoonfit ────────────────────
    print("① 비트 동일성 — hand 축이 s11의 platoonfit와 같은가")
    try:
        import platoonfit
        old = platoonfit.fit_platoon(tr, TARGET_COL, CTX_K)
        new = next(a for a in ctx if a["name"] == "hand")
        check("keys 동일", np.array_equal(old["keys"], new["keys"]),
              f"{len(new['keys']):,}개")
        d = float(np.abs(old["delta"] - new["delta"]).max())
        check("delta 동일", d == 0.0, f"max|diff| = {d:.3e}")
    except ImportError:
        print("  ⏭  platoonfit.py 없음 (이미 정리됨). 이 검사는 리팩터 시점에 통과했다.")

    # ── ② 측정 재현 ────────────────────────────────────────────────
    print("\n② 측정 재현 — logit_adjust가 ctxsplit2와 같은 BSS를 주는가")
    cached = foldcache.load([FOLD], verbose=False)
    yv = cached[FOLD][2]
    z = (1 - MLP_ALPHA) * cached[FOLD][0] + MLP_ALPHA * cached[FOLD][1]
    assert len(yv) == len(va), "캐시 fold와 valid 행 수가 다르다"

    art = {"ctx": ctx}
    z_adj, hits = logit_adjust(va, art)
    print(f"     적중률 " + "  ".join(f"{k} {v:.1%}" for k, v in hits.items())
          + f"   보정 평균 {z_adj.mean():+.5f}  산포 {z_adj.std():.5f}")

    # hand만 / hand+count4 각각의 BSS. ctxsplit2.log의 값과 대조한다.
    hand_only = next(a for a in ctx if a["name"] == "hand")
    z_h, _ = logit_adjust(va, {"ctx": [hand_only]})
    b_plain = bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), yv.mean()))
    b_hand = bss(yv, shift_to_prior(1 / (1 + np.exp(-(z + z_h))), yv.mean()))
    b_both = bss(yv, shift_to_prior(1 / (1 + np.exp(-(z + z_adj))), yv.mean()))
    print(f"     블렌드만       {b_plain:8.1f}")
    print(f"     + hand         {b_hand:8.1f}   ({b_hand - b_plain:+.2f})")
    print(f"     + count4       {b_both:8.1f}   ({b_both - b_hand:+.2f})  ← 증분")
    check("hand 이득이 §7.36 기록(+20.03)과 일치", abs((b_hand - b_plain) - 20.03) < 0.1,
          f"{b_hand - b_plain:+.2f}")
    check("count4 증분이 §7.37 측정(+9.74)과 일치", abs((b_both - b_hand) - 9.74) < 0.1,
          f"{b_both - b_hand:+.2f}")

    # ── ③ 행 독립성 ────────────────────────────────────────────────
    print("\n③ 행 독립성 — 조각 호출 == 통짜 호출")
    idx = np.arange(len(va))
    chunks = [logit_adjust(va.iloc[i:i + 5000], art)[0] for i in range(0, len(va), 5000)]
    d1 = float(np.abs(np.concatenate(chunks) - z_adj).max())
    check("조각 5000행 vs 통짜", d1 == 0.0, f"max|diff| = {d1:.3e}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(va))
    z_sh, _ = logit_adjust(va.iloc[perm], art)
    inv = np.empty_like(perm)
    inv[perm] = idx
    d2 = float(np.abs(z_sh[inv] - z_adj).max())
    check("셔플 후 복원", d2 == 0.0, f"max|diff| = {d2:.3e}")

    print("\n" + ("=" * 60))
    if FAIL:
        raise SystemExit(f"❌ 실패 {len(FAIL)}건: {FAIL}")
    print("✅ 전부 통과. 배포 경로가 측정 경로와 같다.")


if __name__ == "__main__":
    main()
