"""시즌 경계 믹스 변화의 **이질성** — "선수마다 다르다"를 직접 검정한다.

§7.52 / §7.53이 답한 것과 답하지 않은 것
    답함  경계가 **평균적으로** 특별한가        → 아니다 (경과 시간이 전부, §7.53 M5b)
    답함  누가 바뀔지 **미리** 알 수 있나        → 없다 (형질 아님, +0.042)
    ❌    경계에서 **크게 바뀐 놈과 안 바뀐 놈이 갈리는가**  ← 사용자 주장. 여기서 잰다

    앞의 둘은 **평균**에 대한 진술이고 사용자 주장은 **분산**에 대한 진술이다.
    평균 효과가 없어도 개체 간 산포는 클 수 있다.

⚠️ M3(형질 검정)은 **잘못된 관문이었다**
    "누가 바뀔지 예측"이 필요한 것은 Trackman으로 배포할 때뿐이다.
    `asofdelta`가 2025 행에서 그 투수의 **올해 믹스를 직접 준다**
    (`asof_pitcher_pitchmix_n × rate` 차분, 오프셋은 2024 종료 시점).
    예측이 아니라 **관측**이면 형질일 필요가 없다 — 상태를 상태로 읽으면 된다.

검정 설계 — 반쪽 쪼개기 (split-half)
    시즌 s의 투구를 무작위 반으로 (A1, A2), 시즌 s+1도 (B1, B2).
        x = log χ²/df(A1, B1)      y = log χ²/df(A2, B2)
    둘은 **같은 투수의 같은 경계**를 겹치는 공 없이 두 번 잰 값이다.

        corr(x, y) ≈ 0   →  전부 표집 노이즈. "선수마다 다르다"가 거짓
        corr(x, y) > 0   →  진짜 개체차. 그 값이 곧 **신뢰도**(신호 분산 비율)

    대조군: 짝을 섞는다(A2를 **다른 투수의** B2와). 0이 나와야 설계가 맞다.

⚠️ 이 검정이 통과해도 채택 근거가 아니다. "측정 가능한 개체차가 있다"까지고,
   그것이 제구로 이어지는지는 별도다 (§7.52 P3 +0.035, §7.53 M4 +0.033 — 둘 다
   **조절변수** 틀로만 쟀다. **주효과** 틀은 아직 안 쟀다).

실행:  .venv/Scripts/python.exe -u src/mixhet.py 2>&1 | tee experiments/mixhet.log
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

from config import DATA_DIR, MODEL_DIR, SEED
from mixvol import chi2_over_df

GROUPS = ["fastball", "breaking", "offspeed"]
MIN_N = 200          # 반으로 쪼개므로 한쪽당 100구는 남아야 한다


def main():
    rng = np.random.default_rng(SEED)

    m = pd.read_csv(MODEL_DIR / "pitcher_id_map.csv")
    tm = pd.read_csv(DATA_DIR / "trackman_history.csv",
                     usecols=["pitcher_trackman_id", "season", "pitch_type_group"])
    tm = tm.merge(m, on="pitcher_trackman_id", how="inner")
    tm["grp"] = tm["pitch_type_group"].astype(str).str.strip().str.lower()
    tm = tm[tm.grp.isin(GROUPS)].copy()
    tm["gi"] = pd.Categorical(tm.grp, categories=GROUPS).codes
    tm["half"] = rng.integers(0, 2, len(tm))          # 투구 단위 무작위 절반

    # (투수, 시즌, 절반) → 구종 카운트 3벡터
    c = (tm.groupby(["pitcher_id", "season", "half", "gi"]).size()
         .unstack(fill_value=0).reindex(columns=range(3), fill_value=0))
    c["n"] = c.sum(axis=1)
    c = c.reset_index()

    tot = c.groupby(["pitcher_id", "season"])["n"].sum().rename("ntot").reset_index()
    c = c.merge(tot, on=["pitcher_id", "season"])
    c = c[c.ntot >= MIN_N]

    wide = c.pivot_table(index=["pitcher_id", "season"], columns="half",
                         values=[0, 1, 2], fill_value=0)
    wide.columns = [f"h{h}_{g}" for g, h in wide.columns]
    wide = wide.reset_index().sort_values(["pitcher_id", "season"])

    nxt = wide.groupby("pitcher_id").shift(-1)
    ok = (nxt["season"] == wide["season"] + 1).to_numpy()
    A = {h: wide[[f"h{h}_{g}" for g in range(3)]].to_numpy(dtype=np.float64)[ok]
         for h in (0, 1)}
    B = {h: nxt[[f"h{h}_{g}" for g in range(3)]].to_numpy(dtype=np.float64)[ok]
         for h in (0, 1)}
    pid = wide.pitcher_id.to_numpy()[ok]
    s0 = wide.season.to_numpy()[ok]

    x = np.array([chi2_over_df(A[0][i], B[0][i]) for i in range(len(pid))])
    y = np.array([chi2_over_df(A[1][i], B[1][i]) for i in range(len(pid))])
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[good]), np.log(y[good])

    print(f"{'='*70}\n[H1] ★ 같은 경계를 두 번 잰다 — 투수마다 진짜로 다른가"
          f"\n{'='*70}")
    print(f"  경계 {good.sum():,}개 / 투수 {len(np.unique(pid[good])):,}명"
          f"   (양 시즌 {MIN_N}구 이상)")
    r = np.corrcoef(lx, ly)[0, 1]
    rs = pd.Series(lx).corr(pd.Series(ly), method="spearman")
    print(f"\n  corr(반쪽1, 반쪽2)  Pearson {r:+.3f}   Spearman {rs:+.3f}")

    # 대조군 — 짝을 섞는다. 설계가 맞으면 0이 나와야 한다
    perm = rng.permutation(good.sum())
    rc = np.corrcoef(lx, ly[perm])[0, 1]
    print(f"  대조군(짝 섞음)      Pearson {rc:+.3f}   ← 0 근처여야 설계가 맞다")

    if r > 0:
        # Spearman-Brown: 반쪽 신뢰도 → 전체 측정의 신뢰도
        rel = 2 * r / (1 + r)
        print(f"\n  → 개체차가 실재한다. 전체 측정 신뢰도(Spearman-Brown) {rel:.3f}")
        print(f"     = 관측된 변화량 산포의 {rel*100:.0f}%가 진짜 개체차,"
              f" {100-rel*100:.0f}%가 표집 노이즈")

    # ── H2. 크기 — '들썩이는 애'가 실제로 얼마나 다른가 ────────────
    print(f"\n{'='*70}\n[H2] 크기 — 노이즈를 걷어낸 뒤 상·하위가 얼마나 벌어지나\n{'='*70}")
    # 반쪽 평균으로 노이즈를 줄인 추정치로 순위를 매기고, 직구 비중 변화로 본다
    both = np.exp((lx + ly) / 2)
    dff = np.array([A[0][i].sum() and (A[0][i] + A[1][i])[0] / (A[0][i] + A[1][i]).sum()
                    - (B[0][i] + B[1][i])[0] / (B[0][i] + B[1][i]).sum()
                    for i in range(len(pid))])[good]
    q = pd.DataFrame({"pid": pid[good], "s0": s0[good], "chi2df": both,
                      "d_ff": dff, "half1": x[good], "half2": y[good]})
    ter = pd.qcut(q.chi2df, 4, labels=["안정", "중하", "중상", "들썩"])
    print(q.groupby(ter, observed=True).agg(
        경계수=("chi2df", "size"), chi2df중앙=("chi2df", "median"),
        직구비중변화_절대중앙=("d_ff", lambda v: np.median(np.abs(v))),
        직구비중변화_p90=("d_ff", lambda v: np.quantile(np.abs(v), .9))).to_string())
    print(f"\n  전체 |직구비중 변화| : 중앙 {np.median(np.abs(q.d_ff)):.3f}"
          f"   p90 {np.quantile(np.abs(q.d_ff), .9):.3f}"
          f"   p99 {np.quantile(np.abs(q.d_ff), .99):.3f}")

    out = Path(__file__).parent.parent / "experiments" / "mixhet.csv"
    q.to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
