"""§7.59 — `r3_ent`는 왜 부호가 뒤집혔나 (s18: LB −11.01).

홀드아웃 게이트를 **전부** 통과한 축이다:
    양 fold 이득 +21.13 / +12.97 · 크기비 1.63 · 고원 25배 · 대조군 단조 음수
    잔차 t +8.72 / +5.99 (부호 일치) · 얼리는 파라미터 6개 (채택 축 중 최소)

그런데 파라미터 수 법칙의 예측(+11~+14)을 **부호째** 반증했다.
채택된 gap 축들과 무엇이 달랐는지 세 가지를 묻는다.

Q1  **중복** — `r3_ent`는 모델이 이미 갖고 있는 변수인가?
    배포 피처에 `pitchmix_entropy`가 있다 (features.py, 스무딩된 믹스율의 엔트로피).
    거의 같은 변수라면 이 축은 **새 정보가 아니라 기존 피처의 재가중**이다.
    gap 축은 그 반대다 — 시즌 진행분 분해는 어떤 컬럼에도 없다.

Q2  **레벨 드리프트** — 값 자체가 리그 차원에서 흘러가는가?
    gap은 **차분**이라 리그 레벨 이동이 상쇄된다. 엔트로피는 **레벨**이다.
    mu/sd를 학습 시즌에 얼려두고 2025에 적용하면 그 흐름이 로짓 편향으로 들어간다.

Q3  ★ **관계 안정성** — 시즌마다 같은 모양인가?
    이게 진짜 게이트다. §7.58은 **점수(ΔBSS)의 부호**만 두 fold에서 확인했다.
    memory: "부호 일치는 점수가 아니라 관계에 걸어라" — 두 fold 다 이득 양수인데
    잔차 회귀 부호가 반대여서 LB −13.6이 났던 전례가 있다.
    여기서는 모델 없이 **원시 조건부 관계**(십분위별 y 평균, 시즌 레벨 제거)를
    6시즌 전부에 대해 재고, gap 축과 나란히 놓는다.

실행:  .venv/Scripts/python.exe -u src/mixdiag.py 2>&1 | tee experiments/mixdiag.log
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
import features as F
import mixent
from config import ASOF_STRENGTH, SEED, TARGET_COL, TRAIN_CSV
from data import load_csv

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
NBIN = 10


def shape(y, v, nbin=NBIN):
    """십분위별 y 평균에서 **전체 평균을 뺀 모양.** 레벨이 아니라 구조만 남는다.

    memory: "원시 전이 통계는 레벨과 구조를 섞는다" — 레벨 이동은 prior shift가
    흡수하므로, 전이 여부를 묻는 대상은 이 모양이다.
    """
    # ⚠️ 분위수 경계로 자르면 **동점이 많은 축에서 빈 통이 생긴다** (gap은 미등록
    #    행이 전부 같은 값이라 실제로 그랬고, 상관이 통째로 nan이 됐다).
    #    순위를 등분해 자르면 통 크기가 항상 같다.
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    b = np.minimum((r * nbin) // len(v), nbin - 1)
    m = np.array([y[b == i].mean() for i in range(nbin)])
    return m - m.mean()


def slope(y, v):
    """표준화된 v에 대한 y의 단순 회귀 기울기 (레벨 무관)."""
    vv = (v - v.mean()) / (v.std() or 1.0)
    return float(np.cov(vv, y)[0, 1])


def main():
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)

    # 축들을 **학습 창(≤2022)에서 fit** 해서 전 시즌에 적용한다.
    # 배포와 같은 상황을 만들려는 것이다 — 상수는 과거에서 얼리고 미래에 쓴다.
    tr22 = s <= 2022
    mix = mixent.fit_mixent(df[tr22], [("r3_ent", 1.0)])[0]
    asof = asofdelta.fit_asof_axes(df[tr22],
                                   [("gap_success", 1.0), ("gap_bsuccess", 1.0)],
                                   ASOF_STRENGTH)
    # ⚠️ 오프셋만은 **전 시즌**에서 만든다. gap은 "그 시즌 시작 시점의 누적"을 빼는
    #    것이라, ≤2022 오프셋으로 2023/2024 행을 보면 조회가 전부 실패해 gap≈0이
    #    된다 (초안이 그래서 기울기 0.0000·상관 nan을 뱉었다).
    #    여기서 묻는 것은 점수가 아니라 **관계가 시즌마다 같은 모양인가**이므로
    #    각 시즌의 gap이 제대로 계산되어야 한다. 표준화 상수(mu/sd)는 ≤2022 그대로다.
    off_all = asofdelta.fit_offsets(df)
    for a in asof:
        o = off_all[a["n_col"]]
        a["keys"], a["n_off"] = o["keys"], o["n_off"]
        a["k_off"] = o["k_off"][a["r_col"]]

    V = {"r3_ent ❌LB−11": F.mixent_lookup(df, mix)}
    for a in asof:
        V[a["name"] + " ✅채택"] = F.asof_lookup(df, a)

    # ── 게이트 검증용 대조군 ────────────────────────────────────────
    # 같은 gap 계열에서 **기각된** 축들. 게이트가 사후 이야기가 아니라면
    # 채택된 둘과 기각된 셋을 같은 방향으로 갈라야 한다.
    for nm, rate, ncol in (("gap_reverse ❌기각", "asof_pitcher_reverse_rate",
                            "asof_pitcher_n"),
                           ("gap_middle ❌기각", "asof_pitcher_middle_rate",
                            "asof_pitcher_n"),
                           ("gap_ball ❌기각", "asof_pitcher_ball_rate",
                            "asof_pitcher_n")):
        o = off_all[ncol]
        idc = "batter_id" if ncol.startswith("asof_batter") else "pitcher_id"
        a = dict(id=idc, n_col=ncol, r_col=rate, keys=o["keys"], n_off=o["n_off"],
                 k_off=o["k_off"][rate], mu=0.0, sd=1.0,
                 league=float(np.nanmean(df[rate].to_numpy(dtype=np.float64))),
                 strength=ASOF_STRENGTH)
        raw = F.asof_lookup(df, a)
        V[nm] = (raw - raw[tr22].mean()) / (raw[tr22].std() or 1.0)
    # 배포 피처 쪽 엔트로피 (스무딩 버전). 중복 여부를 보려면 같은 행에서 재야 한다.
    art = F.fit_artifacts(df[tr22])
    V["pitchmix_entropy(배포피처)"] = F.build_features(df, art)["pitchmix_entropy"] \
        .to_numpy(dtype=np.float64)

    print("=" * 88)
    print("Q1  중복 — r3_ent는 모델이 이미 갖고 있는 변수인가")
    print("=" * 88)
    for nm in [k for k in V if "배포피처" not in k]:
        cc = np.corrcoef(V[nm], V["pitchmix_entropy(배포피처)"])[0, 1]
        print(f"  corr({nm:<20}, 배포 pitchmix_entropy) = {cc:+.4f}")
    print("  → 1에 가까우면 이 축은 '새 정보'가 아니라 **기존 피처의 재가중**이다.")

    print("\n" + "=" * 88)
    print("Q2  레벨 드리프트 — 표준화 상수를 ≤2022에서 얼린 뒤 시즌별 mean(v)")
    print("=" * 88)
    print(f"  {'축':<22}" + "".join(f"{yr:>9}" for yr in SEASONS))
    for nm in [k for k in V if "배포피처" not in k]:
        row = [float(V[nm][s == yr].mean()) for yr in SEASONS]
        print(f"  {nm:<22}" + "".join(f"{r:>+9.3f}" for r in row))
    print("  → gap은 **차분**이라 리그 레벨 이동이 상쇄된다. 엔트로피는 레벨이다.")
    print("     2022→2024 이동폭이 크면 2025 외삽분이 그대로 로짓 편향이 된다.")

    print("\n" + "=" * 88)
    print("Q3 ★ 관계 안정성 — 시즌별 원시 기울기 (모델 없음, 레벨 제거)")
    print("=" * 88)
    print(f"  {'축':<22}" + "".join(f"{yr:>9}" for yr in SEASONS) + "   추세")
    for nm in [k for k in V if "배포피처" not in k]:
        row = [slope(y[s == yr], V[nm][s == yr]) for yr in SEASONS]
        # 마지막 3시즌의 선형 추세 — 2025로 외삽했을 때의 방향
        tail = np.polyfit([0, 1, 2], row[-3:], 1)[0]
        nxt = row[-1] + tail
        print(f"  {nm:<22}" + "".join(f"{r:>+9.4f}" for r in row)
              + f"   →2025 {nxt:+.4f}")

    # ⚠️ **노이즈 천장부터 잰다.** 시즌 간 상관이 낮은 것은 두 가지를 뜻할 수 있다 —
    #    구조가 안 유지되거나, 애초에 구조를 측정하지 못했거나.
    #    r3_ent는 기울기가 gap의 절반이라 감쇠를 의심해야 한다.
    #    memory: "비율 조건에는 노이즈 게이트가 먼저다".
    #    시즌 안에서 행을 반으로 갈라 같은 모양을 두 번 재고 Spearman–Brown으로 올린다.
    rng = np.random.default_rng(SEED)
    print("\n  십분위 모양의 시즌 간 상관 (구조가 유지되는가)")
    print(f"  {'축':<22}" + "".join(f"{a}-{b}".rjust(11)
                                    for a, b in zip(SEASONS, SEASONS[1:]))
          + "   최근3 최소   천장(신뢰도)   비율")
    for nm in [k for k in V if "배포피처" not in k]:
        sh, rel = [], []
        for yr in SEASONS:
            m = s == yr
            yy, vv = y[m], V[nm][m]
            sh.append(shape(yy, vv))
            h = rng.random(len(yy)) < 0.5
            r = float(np.corrcoef(shape(yy[h], vv[h]), shape(yy[~h], vv[~h]))[0, 1])
            rel.append(2 * r / (1 + r) if r > -1 else np.nan)   # Spearman–Brown
        cs = [float(np.corrcoef(sh[i], sh[i + 1])[0, 1]) for i in range(len(sh) - 1)]
        lo = min(cs[-3:])
        ceil = float(np.nanmean(rel[-3:]))
        print(f"  {nm:<22}" + "".join(f"{v:>+11.3f}" for v in cs)
              + f"{lo:>+11.3f}{ceil:>14.3f}{lo / ceil if ceil else np.nan:>8.2f}")

    print("\n" + "=" * 88)
    print("읽는 법")
    print("=" * 88)
    print("  Q1이 크고 Q3가 불안정하면 — 이 축은 새 정보가 아니라 검증 시즌에 국한된")
    print("  **오차 보정**이었다. 홀드아웃은 그 오차를 보므로 항상 이득으로 보인다.")
    print("  §7.58이 '정보가 아니라 전달이 문제다'(§7.56과 같은 그림)로 읽은 것이")
    print("  정확히 뒤집힌 해석일 수 있다.")


if __name__ == "__main__":
    main()
