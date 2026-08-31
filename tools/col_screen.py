"""배포 78피처 **전부**를 컬럼 선형 δ 축 후보로 훑는다 — `hj16` 층 §7.91 잔차 지도.

왜
    최근 이득이 전부 **로짓공간 선형축**이다 (`art['col']`, §7.223·§7.226 = +7.30).
    통로는 `z += β·(x−μ)/σ` 하나인데 배포 컬럼이 78개다. 원우는 fold 하네스로
    한 번에 하나씩 묻는다 — 여기서는 **전 컬럼을 한 번에** 훑어 후보 목록을 만든다.

    ⚠️ 잔차가 챔피언 **in-sample** 이다. 그래서 이 출력은 **채택 근거가 아니라
       후보 목록**이다 (라운드 25: 홀드아웃은 발굴 전용). 판정은 원우 하네스가 한다.

수식 — 라우팅·C1 과 같은 틀
    dz = β·v,  v = (x−μ)/σ          dp = PROB_SCALE·p(1−p)·dz
    g  = PROB_SCALE·p(1−p)·v (중심화)
    ΔScore(β) = −K[2β·M + β²·G],  M = mean(g(p−y)),  G = mean(g²)
    β* = −M/G,   최대이득 = K·M²/G

G0 — 이미 배포된 두 축(`middle_rate_sm`·`success_rate`)의 **잔여 이득이 0 근처**여야
     한다. 배포 β 가 최적이면 더 짤 게 없다. 아니면 이 화면이 틀린 것이다.

교차 시즌 — 한 해에서 고른 β* 를 다른 해에서 채점한다 (원우 §7.227 의 교차 채점).
"""
import argparse, os, subprocess, sys, zipfile, tempfile
import joblib, numpy as np, pandas as pd

TRAIN = "/Users/hyejin/Aimers_9th/data/train.csv"
K = 1e5 / (0.486987 * (1 - 0.486987))
SCALE, CENTER = 1.05, 0.4801


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True, help="풀어 둔 챔피언 zip")
    ap.add_argument("--pred", required=True, help="그 챔피언의 전 시즌 train 예측 .npy")
    ap.add_argument("--shift", type=float, required=True, help="그 zip 의 PROB_SHIFT")
    ap.add_argument("--fit", type=int, default=2023)
    ap.add_argument("--score", type=int, default=2024)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    sys.path.insert(0, a.zip_dir)
    from features import build_features
    b = joblib.load(os.path.join(a.zip_dir, "model", "artifacts.joblib"))
    art, cols = b["artifacts"], list(b["feature_cols"])
    deployed = {x["name"] for x in (art.get("col") or [])}
    print(f"배포 컬럼 {len(cols)}개 · 이미 col 축인 것 {sorted(deployed)}")

    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    y = df["control_success"].to_numpy(np.float64)
    s = df["season"].to_numpy()
    p = np.load(a.pred)
    p_raw = (p - a.shift - CENTER) / SCALE + CENTER
    J = SCALE * p_raw * (1 - p_raw)              # dp/dz
    resid = p - y
    print(f"전 시즌 {len(df):,}행 · 잔차 rms {np.sqrt((resid**2).mean()):.5f}")

    src = df.drop(columns=["control_success"])
    X = build_features(src, art)
    if art.get("use_player_ids"):
        for c in ("pitcher_id", "batter_id"):
            X[c] = src[c].to_numpy(np.float32)
    print(f"피처 생성 완료 {X.shape}")

    mf, ms = s == a.fit, s == a.score
    rows = []
    for c in cols:
        x = np.asarray(X[c], np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 1000 or np.nanstd(x[ok]) == 0:
            continue
        mu, sd = float(x[ok].mean()), float(x[ok].std())
        v = (np.where(ok, x, mu) - mu) / sd
        g = J * v
        def stat(m):
            gm = g[m] - g[m].mean()
            G = float((gm ** 2).mean()); M = float((gm * resid[m]).mean())
            return M, G, K * M * M / G, -M / G
        Mf, Gf, gain_f, bf = stat(mf)
        Ms, Gs, gain_s, bs = stat(ms)
        cross = -(2 * bf * Ms * K + bf * bf * K * Gs)      # fit 의 β* 를 score 에서 채점
        rows.append((gain_s, c, gain_f, gain_s, bf, bs, cross, K * Gs))

    rows.sort(reverse=True)
    print(f"\n{'컬럼':44s} {'f'+str(a.fit):>9} {'f'+str(a.score):>9} "
          f"{'β*fit':>9} {'β*scr':>9} {'교차':>9} {'C':>8}")
    print("-" * 104)
    for gs, c, gf, gs2, bf, bs, cr, C in rows[:a.top]:
        tag = "  ← 배포" if c in deployed else ""
        print(f"{c:44s} {gf:+9.2f} {gs2:+9.2f} {bf:+9.5f} {bs:+9.5f} {cr:+9.3f} {C:8.1f}{tag}")

    print(f"\n=== G0 — 이미 배포된 축의 잔여 이득 (0 근처여야 한다) ===")
    for gs, c, gf, gs2, bf, bs, cr, C in rows:
        if c in deployed:
            print(f"  {c:44s} f{a.fit} {gf:+8.3f}  f{a.score} {gs2:+8.3f}  β* {bs:+.5f}")

    print(f"\n=== 관문 통과 (양 fold 양수 · 교차 채점 양수 · 미배포) ===")
    ok = [r for r in rows if r[2] > 0 and r[3] > 0 and r[6] > 0 and r[1] not in deployed]
    print(f"  {len(ok)} / {len(rows)}")
    for gs, c, gf, gs2, bf, bs, cr, C in ok[:15]:
        print(f"  {c:44s} f{a.fit} {gf:+8.2f}  f{a.score} {gs2:+8.2f}  교차 {cr:+7.3f}  β* {bf:+.5f}")


if __name__ == "__main__":
    main()
