"""§7.118-A — δ 를 확률차 대신 **로짓차**로 재정의하면 나아지는가 (RESUME #2).

기전 (코드로 확인되는 스케일 불일치)
    ctxfit.fit_ctx 의 δ = shrink·(rate_pc − rate_p) 는 **확률의 차이**인데,
    predict 는 z += β·δ 로 **로짓에** 더한다. 확률 이동 Δp 를 재현하는 데 필요한
    로짓 이동은 Δp / (p(1−p)) 라서, 기저 성공률이 극단인 투수일수록 같은 δ 가
    체계적으로 덜 들어간다. 이 모델의 p 범위는 [0.31, 0.67] → p(1−p) 0.21~0.25,
    왜곡 상한 ~15%. 교정: δ_logit = shrink·(logit(rate_pc) − logit(rate_p)).

왜 재포장이 아닌가
    §7.43 의 "δ 채널 포화"는 **축**(어떤 상황 스플릿이 신호인가)에 대한 선언이다.
    이것은 채택돼 LB 전이가 실측된 두 축(hand +5.93, count4 +3.08)의 **추정량** 교정.

사전 등록 (§7.118-A)
    하네스: 확률차 δ 를 이 코드 경로로 재적합하면 배포 Xax 컬럼과 allclose 여야
      하고, 기준 BSS 는 807.2/924.5 재현 (deploy29 가 assert).
    관문: 양 fold Δ>0 & 클러스터 t≥2 & 크기비 ≤3 & β 고원. 하나라도 미달 → 기각.
    사전 예측: corr(δ_p, δ_logit) 이 매우 높아 Δ +1~3 노이즈 대역
      — "판정 불가"로 죽을 공산이 가장 크다. 그 경우 제출하지 않는다.

실행:  .venv/Scripts/python.exe -u src/ctxlogit.py 2>&1 | tee experiments/ctxlogit.log
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
from config import CTX_AXES, CTX_HALF_LIFE, CTX_K, TARGET_COL
from data import recency_weight
from features import CTX_CODERS, ctx_lookup

EPS = 1e-3          # 셀 성공률 로짓 클립 (작은 셀은 shrink 가 이미 누른다)
GRID = {"hand":   [0.25, 0.35, 0.45, 0.55, 0.70, 0.90],
        "count4": [0.15, 0.22, 0.30, 0.40, 0.55, 0.75]}
DEP_BETA = dict(CTX_AXES)      # hand 1.875 / count4 1.25


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_ctx_both(train_df, k, weights):
    """ctxfit.fit_ctx 와 같은 절차로 확률차/로짓차 δ 를 **동시에** 만든다.

    확률차 쪽은 배포 코드(ctxfit)와 비트 동일해야 한다 — 하네스 검증 대상.
    """
    y = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    pid = train_df["pitcher_id"].to_numpy(dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)

    upid, invp = np.unique(pid, return_inverse=True)
    sw_p = np.bincount(invp, weights=w)
    rate_p = np.bincount(invp, weights=w * y) / np.maximum(sw_p, 1e-12)

    out = {}
    for name in ("hand", "count4"):
        code, m = CTX_CODERS[name](train_df)
        key = pid * m + code
        u, inv = np.unique(key, return_inverse=True)
        sw = np.bincount(inv, weights=w)
        sw2 = np.bincount(inv, weights=w * w)
        n_eff = np.where(sw2 > 0, sw * sw / np.maximum(sw2, 1e-12), 0.0)
        t = np.bincount(inv, weights=w * y)
        rc = t / np.maximum(sw, 1e-12)
        rp = rate_p[np.searchsorted(upid, u // m)]
        shrink = n_eff / (n_eff + k)
        out[name] = {
            "prob":  {"name": name, "beta": 1.0, "keys": u,
                      "delta": (shrink * (rc - rp)).astype(np.float64)},
            "logit": {"name": name, "beta": 1.0, "keys": u,
                      "delta": (shrink * (_logit(rc) - _logit(rp))).astype(np.float64)},
        }
    return out


def main():
    df, s, y = D.load_df()
    print(f"train {df.shape}   K={CTX_K}   배포 β {DEP_BETA}   지문 {D.KEY}\n")

    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        tr = df[s < fold]
        w = recency_weight(tr["season"].to_numpy(), CTX_HALF_LIFE,
                           ref_season=int(s[s < fold].max()))
        tabs = fit_ctx_both(tr, CTX_K, w)

        # 행 값 (clean 검증 행)
        V = {}
        for name in ("hand", "count4"):
            V[(name, "prob")] = ctx_lookup(st["d"], tabs[name]["prob"])
            V[(name, "logit")] = ctx_lookup(st["d"], tabs[name]["logit"])

        # ── 하네스: 확률차 재적합 == 배포 Xax 컬럼 ─────────────────────
        print("=" * 92)
        print(f"fold {fold}   기준 BSS {st['base']:.1f}")
        print("=" * 92)
        for name in ("hand", "count4"):
            xj = st["Xax"][:, st["names"].index(name)]
            ok = np.allclose(V[(name, "prob")], xj, atol=1e-10)
            c = np.corrcoef(V[(name, "prob")], V[(name, "logit")])[0, 1]
            # 로짓판/확률판 스케일 비 (std 기준) — 등가 β 환산 근거
            ratio = V[(name, "logit")].std() / max(V[(name, "prob")].std(), 1e-12)
            print(f"  {name:<8} 하네스(확률차==배포) {'OK' if ok else '★★ 불일치'}   "
                  f"corr(δ_p, δ_logit) {c:.5f}   스케일비 {ratio:.3f}   "
                  f"등가 β ≈ {DEP_BETA[name] / ratio:.3f}")
            assert ok, "확률차 재적합이 배포와 다르다 — 이후 숫자 전부 무효"

        # ── 1D: 한 축만 로짓판으로 교체 (다른 축은 배포 유지) ──────────
        for name in ("hand", "count4"):
            xj = st["Xax"][:, st["names"].index(name)]
            drop = -DEP_BETA[name] * xj
            row = []
            for b in GRID[name]:
                row.append(D.rescore(st, drop + b * V[(name, "logit")]))
            print(f"\n  {name} → 로짓판 (β 스윕):")
            print("    " + " ".join(f"{b:>7}" for b in GRID[name]))
            print("    " + " ".join(f"{g:>+7.2f}" for g in row))
            bi = int(np.argmax(row))
            t = D.cluster_t(st, drop + GRID[name][bi] * V[(name, "logit")])
            print(f"    최적 β={GRID[name][bi]}  Δ{row[bi]:+.2f}  클러스터 t {t:+.2f}")

        # ── 2D: 두 축 다 교체 ─────────────────────────────────────────
        drop_all = (- DEP_BETA["hand"] * st["Xax"][:, st["names"].index("hand")]
                    - DEP_BETA["count4"] * st["Xax"][:, st["names"].index("count4")])
        print(f"\n  두 축 동시 교체 (행=β_hand, 열=β_count4):")
        print("        " + " ".join(f"{b:>7}" for b in GRID["count4"]))
        best = (-1e18, None)
        for bh in GRID["hand"]:
            row = []
            for bc in GRID["count4"]:
                g = D.rescore(st, drop_all + bh * V[("hand", "logit")]
                              + bc * V[("count4", "logit")])
                row.append(g)
                if g > best[0]:
                    best = (g, (bh, bc))
            print(f"  {bh:>6}" + " ".join(f"{g:>+7.2f}" for g in row))
        (bh, bc) = best[1]
        t = D.cluster_t(st, drop_all + bh * V[("hand", "logit")]
                        + bc * V[("count4", "logit")])
        print(f"  최적 ({bh}, {bc})  Δ{best[0]:+.2f}  클러스터 t {t:+.2f}")

    print("\n" + "=" * 92)
    print("판정: 양 fold Δ>0 & t≥2 & 크기비 ≤3 & 고원 — 사전 예측은 '판정 불가 사망'.")
    print("최적점은 fold별 argmax 라 낙관 편향이 있다 — 채택 논의 시 밴드 규칙으로 재선정.")


if __name__ == "__main__":
    main()
