"""δ 축 2차 스크린 — 교차축 · 타자 쪽 · recency 가중 (학습 없음)

§7.36의 전수 조사(`ctxsplit.py`)가 남긴 구멍 셋을 메운다.

  ① **교차축.** 9축을 **단독으로만** 쟀다. 채택된 둘(hand · count4)의 **교차**는
     안 쟀다. §7.15가 "명시적 상호작용을 만든다면 손잡이에서 시작"이라고 적어두고
     실행하지 않은 항목이기도 하다.
     ⚠️ 위험이 이미 알려져 있다 — `count12`(12칸)는 셀당 표본이 1/3로 줄어
     **과전이에 부러졌다**. 교차축은 칸이 더 많다. 칸 수와 셀당 표본을 같이 본다.

  ② **타자 쪽.** 9축이 **전부 투수 × 상황**이었다. 타자 쪽 `asof_*`는 3컬럼뿐이라
     (n · success_rate · middle_rate) 오히려 **가장 덜 모델링된 엔티티**다.
     δ(타자, 투수손잡이) = 받는 쪽 플래툰, δ(타자, 카운트) = 타자별 카운트 반응.

  ③ **recency 가중.** `ctxfit`은 δ를 전 시즌 **균등**으로 추정한다. 배포 모델은
     hl=0.5로 최근에 4배 가중을 준다. 둘이 어긋나 있다.

기준선이 §7.36 때와 다르다 — **배포 상태 전체**다
    z0 = 블렌드(α) + β_hand·δ_hand + β_count4·δ_count4
    ctxsplit은 count4가 배포 전이라 hand만 얹었다. 지금 그러면 count4 몫이 새 축의
    이득에 섞인다. **재적용 대조군 둘 다 ≈0이 나와야** 나머지 숫자를 믿을 수 있다.

재는 것 (전부 캐시된 z만 — GPU도 학습도 없다)
    칸 / 셀당표본   과전이 위험의 직접 지표 (count12가 여기서 부러졌다)
    적중률          valid에서 δ를 찾을 수 있는 비율
    T1 corr         δ(과거) vs δ(실제) — 스플릿이 시즌을 넘나
    **T2 잔차 t**   배포 모델이 아직 안 쓰는 정보인가  ← 결정적
    T3 β 스윕       실제 BSS 이득 (양 fold + 크기비)

다중검정
    후보 6개를 재고 최댓값을 고르면 그 최댓값은 편향돼 있다. 여기서 살아남은 것은
    **후보일 뿐**이고, 채택은 단독 재측정으로 정한다 (§7.36이 그렇게 했다).

실행:  .venv/Scripts/python.exe -u src/ctxscreen.py 2>&1 | tee experiments/ctxscreen.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
from config import (CTX_AXES, CTX_K, MLP_ALPHA, RECENCY_HALF_LIFE, TARGET_COL,
                    TRAIN_CSV)
from ctxsplit import actual_delta, build_delta, lookup, resid_t
from data import load_csv, recency_weight
from validation import bss, shift_to_prior

FOLDS = [2023, 2024]
K = CTX_K
BETAS = [0.25, 0.5, 1.0, 1.5, 2.0]


def build_delta_w(ent, code, y, w, m, k):
    """`build_delta`의 가중판. n을 **유효표본수**(가중 합)로 바꾼 것뿐이다.

    수축 n/(n+k)의 n이 가중 합이 되므로, 오래된 시즌만 있는 셀은 자동으로 더
    수축된다 — 이건 부작용이 아니라 의도다 (낡은 스플릿을 덜 믿는다).
    """
    key = ent * m + code
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, weights=w)
    t = np.bincount(inv, weights=w * y)

    uent, inve = np.unique(ent, return_inverse=True)
    n_e = np.bincount(inve, weights=w)
    rate_e = np.bincount(inve, weights=w * y) / n_e
    re = rate_e[np.searchsorted(uent, u // m)]
    return u, (n / (n + k)) * (t / n - re), n


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = df["pitcher_id"].to_numpy(dtype=np.int64)
    bid = df["batter_id"].to_numpy(dtype=np.int64)

    bh = df["batter_hand"].to_numpy(dtype=np.int64)
    ph = df["pitcher_hand"].to_numpy(dtype=np.int64)
    b = df["balls_before"].to_numpy(dtype=np.int64)
    st = df["strikes_before"].to_numpy(dtype=np.int64)
    c4 = (b >= 3).astype(np.int64) + 2 * (st >= 2).astype(np.int64)

    print(f"train {df.shape}   k={K}   배포축 {CTX_AXES}")
    print(f"batter_hand 고유값 {sorted(np.unique(bh))}   "
          f"pitcher_hand 고유값 {sorted(np.unique(ph))}")

    # (이름, 엔티티, 코드, m, 역할)
    CANDS = [
        ("hand(재적용)",    pid, bh,           8,  "대조"),
        ("count4(재적용)",  pid, c4,           4,  "대조"),
        ("hand×count4",     pid, bh * 4 + c4,  32, "후보"),
        ("타자 단독",       bid, np.zeros_like(bid), 1, "대조"),
        ("타자×투수손",     bid, ph,           8,  "후보"),
        ("타자×count4",     bid, c4,           4,  "후보"),
    ]

    cached = foldcache.load(FOLDS, verbose=False)
    store = {}

    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]

        # ── 기준선 = **배포 상태 전체** (hand + count4 둘 다 얹는다) ──
        z0 = z.copy()
        for name, beta in CTX_AXES:
            code, m = (bh, 8) if name == "hand" else (c4, 4)
            u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
            drow, _ = lookup(u, d, pid[va], code[va], m)
            z0 = z0 + beta * drow
        p0 = shift_to_prior(1.0 / (1.0 + np.exp(-z0)), yv.mean())
        base = bss(yv, p0)
        r = yv - p0                                  # 배포 상태의 잔차

        print("\n" + "=" * 104)
        print(f"fold {fold}   train {tr.sum():,} → valid {va.sum():,}"
              f"   기준 BSS(배포상태) {base:.1f}")
        print("=" * 104)
        print(f"  {'축':<15}{'역할':>5}{'칸':>8}{'셀표본':>8}{'적중':>7}"
              f"{'δ산포':>9}{'T1':>7}{'T2(t)':>9}   " +
              "".join(f"β{x:<6.2f}" for x in BETAS))

        for name, ent, code, m, role in CANDS:
            u, d, n = build_delta(ent[tr], code[tr], y[tr], m, K)
            drow, hit = lookup(u, d, ent[va], code[va], m)

            # T1 — 과거 δ 와 검증시즌 실제 δ 의 상관
            ua, da, na = actual_delta(ent[va], code[va], y[va], m)
            j = np.searchsorted(u, ua)
            j = np.clip(j, 0, len(u) - 1)
            ok = (u[j] == ua) & (na >= 30)
            t1 = (float(np.corrcoef(d[j][ok], da[ok])[0, 1])
                  if ok.sum() > 30 else float("nan"))

            _, t2 = resid_t(r, drow)
            gains = []
            for beta in BETAS:
                p = shift_to_prior(1.0 / (1.0 + np.exp(-(z0 + beta * drow))),
                                   yv.mean())
                gains.append(bss(yv, p) - base)
            store.setdefault(name, {})[fold] = (gains, t2, role)

            print(f"  {name:<15}{role:>5}{len(u):>8,}{np.median(n):>8.0f}"
                  f"{hit.mean():>7.1%}{d.std():>9.5f}{t1:>7.3f}{t2:>+9.2f}   "
                  + "".join(f"{g:>+7.2f}" for g in gains))

    # ── 대조군 검산 ────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("대조군 검산 — 재적용 두 축은 **0에 붙어야** 한다 (아니면 기준선이 틀렸다)")
    print("=" * 104)
    for name in ("hand(재적용)", "count4(재적용)"):
        g = [max(store[name][f][0], key=abs) for f in FOLDS]
        print(f"  {name:<15} 최대|이득|  " +
              "  ".join(f"fold{f} {v:+.2f}" for f, v in zip(FOLDS, g)))

    # ── 후보 요약 ─────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("후보 요약 — β별 양 fold 공통 이득 구간이 있는가")
    print("=" * 104)
    print(f"  {'축':<15}" + "".join(f"β{x:<10.2f}" for x in BETAS))
    for name, _, _, _, role in CANDS:
        if role != "후보":
            continue
        row = ""
        for i, beta in enumerate(BETAS):
            a, bb = (store[name][f][0][i] for f in FOLDS)
            mark = "✓" if (a > 0 and bb > 0) else " "
            row += f"{a:>+5.1f}/{bb:>+5.1f}{mark} "
        print(f"  {name:<15}{row}")
    print("\n  ✓ = 양 fold 동시 양수. 없으면 조건 ②에서 이미 탈락이다.")

    # ── Part B — recency 가중 δ (교체 비교) ──────────────────────
    print("\n" + "=" * 104)
    print(f"Part B — δ 추정에 recency 가중 (hl={RECENCY_HALF_LIFE}). **교체** 비교다")
    print("=" * 104)
    print(f"  {'fold':<8}{'균등(배포)':>12}{'recency 가중':>14}{'차이':>9}")
    for fold in FOLDS:
        tr, va = s < fold, s == fold
        yv = cached[fold][2]
        z = (1 - MLP_ALPHA) * cached[fold][0] + MLP_ALPHA * cached[fold][1]
        w = recency_weight(s[tr], RECENCY_HALF_LIFE, ref_season=int(s[tr].max()))

        out = {}
        for tag, wt in (("균등", None), ("recency", w)):
            zz = z.copy()
            for name, beta in CTX_AXES:
                code, m = (bh, 8) if name == "hand" else (c4, 4)
                if wt is None:
                    u, d, _ = build_delta(pid[tr], code[tr], y[tr], m, K)
                else:
                    u, d, _ = build_delta_w(pid[tr], code[tr], y[tr], wt, m, K)
                drow, _ = lookup(u, d, pid[va], code[va], m)
                zz = zz + beta * drow
            out[tag] = bss(yv, shift_to_prior(1.0 / (1.0 + np.exp(-zz)), yv.mean()))
        print(f"  {fold:<8}{out['균등']:>12.1f}{out['recency']:>14.1f}"
              f"{out['recency']-out['균등']:>+9.2f}")
    print("\n  양 fold 동시 양수가 아니면 조건 ②에서 탈락 — 그대로 둔다.")


if __name__ == "__main__":
    main()
