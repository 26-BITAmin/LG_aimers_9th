"""보조 헤드 확장 전 사전 검사 — ball/strike는 **새 축인가, 기존 축의 함수인가**

왜 이걸 먼저 하는가
    s13이 채택한 보조 헤드는 (몰림, 반대) 2개다. `modelabel`은 같은 방식으로
    (볼, 스트라이크)도 99.93% 복원한다. 헤드를 4개로 늘리는 건 기계가 이미 있으니
    코드로는 한 줄이다. 문제는 **그게 정보를 늘리는가**다.

    ablate39b가 배운 것: 보조 헤드 이득의 상당 부분은 **정규화**다. 무작위 라벨
    (`rand`)로도 fold2023에서 +32.9가 나왔다. 헤드를 2개에서 4개로 늘리면
    정규화는 확실히 늘어난다. 그러니 새 헤드가 **기존 헤드로부터 예측되지 않는
    정보**를 갖고 있지 않으면, 늘린 이득은 전부 정규화이고 fold2024에서 부호가
    뒤집힐 것이다 (`rand`가 정확히 그랬다: +32.9 / −6.6).

    → 25분짜리 GPU 실험을 돌리기 전에 30초짜리 조건부 엔트로피로 게이트한다.
      (memory: "새 피처는 잔차에 먼저 물어라"의 타깃 판)

측정
    ① H(ball | suc, mid, rev)  vs  H(ball)     — 남은 정보량
    ② 3분류 결과(볼/스트라이크/인플레이)에 대해 같은 것
    ③ 셀별 분포 — 어느 조합에서 갈리는가 (기전을 봐야 채택 근거가 된다)
    ④ 카운트에 따라 ball/strike가 mid/rev와 **다르게** 움직이는가
       (§7.40의 착상: 축이 다르려면 카운트 반응이 달라야 한다)

실행:  .venv/Scripts/python.exe -u src/auxinfo.py 2>&1 | tee experiments/auxinfo.log
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

from config import TRAIN_CSV
from data import load_csv
from modelabel import recover


def ent(p):
    """비트 단위 엔트로피. p는 확률 벡터(합 1)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cond_ent(y, g):
    """H(y | g). y는 정수 라벨, g는 정수 그룹."""
    n = len(y)
    tot = 0.0
    for gv in np.unique(g):
        m = g == gv
        cnt = np.bincount(y[m])
        tot += (m.sum() / n) * ent(cnt / m.sum())
    return tot


def main():
    df = load_csv(TRAIN_CSV)
    lab, _, _ = recover(df)
    keys = ("success", "middle", "reverse", "ball", "strike")
    v = {k: np.round(lab[k]) for k in keys}
    ok = ~np.any([np.isnan(v[k]) for k in keys], axis=0)
    # {0,1}에 깨끗이 떨어지는 행만 (부동소수점 잔차가 큰 0.07%는 버린다)
    for k in keys:
        ok &= np.isin(v[k], [0.0, 1.0])
    v = {k: v[k][ok].astype(np.int64) for k in keys}
    n = int(ok.sum())
    print(f"train {df.shape} → 라벨 5종이 모두 깨끗한 행 {n:,} ({ok.mean():.2%})\n")

    suc, mid, rev, ball, stk = (v[k] for k in keys)
    off = 1 - suc - mid - rev            # 세 번째 실패 모드 (겹침 때문에 음수 가능)
    print(f"양성률   suc {suc.mean():.4f}  mid {mid.mean():.4f}  rev {rev.mean():.4f}"
          f"  ball {ball.mean():.4f}  strike {stk.mean():.4f}")
    print(f"         off(=1−suc−mid−rev) 평균 {off.mean():.4f}   음수 {np.mean(off<0):.2%}")
    print(f"         ball & strike 동시 1  {np.mean((ball==1)&(stk==1)):.4%}"
          f"   둘 다 0 (인플레이 추정)  {np.mean((ball==0)&(stk==0)):.4f}")

    # ── ① 기존 3축이 주어졌을 때 ball/strike에 남는 정보 ──────────────────
    g3 = suc + 2 * mid + 4 * rev          # (suc, mid, rev) 조합 코드 — 8셀
    print("\n" + "=" * 92)
    print("① 기존 축 (suc, mid, rev) 가 주어졌을 때 ball/strike에 **남는** 정보")
    print("=" * 92)
    print(f"  {'타깃':<22}{'H(y)':>9}{'H(y|suc,mid,rev)':>20}{'남은 비율':>12}")
    rows = [("ball", ball), ("strike", stk)]
    # 3분류: 0=인플레이, 1=볼, 2=스트라이크
    out3 = np.where(ball == 1, 1, np.where(stk == 1, 2, 0))
    rows.append(("결과 3분류", out3))
    for name, t in rows:
        h = ent(np.bincount(t) / len(t))
        hc = cond_ent(t, g3)
        print(f"  {name:<22}{h:>9.4f}{hc:>20.4f}{hc/h:>11.1%}")
    print("\n  ← '남은 비율'이 크면 ball/strike는 **독립적인 새 축**이다.")
    print("     0에 가까우면 기존 헤드의 함수일 뿐이고, 4헤드는 정규화만 늘린다.")

    # 대조군: mid/rev 자신은 suc만 주어졌을 때 얼마나 남는가
    print("\n  [대조] 이미 채택된 헤드가 같은 잣대에서 얼마였는가:")
    for name, t in (("middle", mid), ("reverse", rev)):
        h = ent(np.bincount(t) / len(t))
        hc = cond_ent(t, suc)
        print(f"  {name:<22}{h:>9.4f}{hc:>20.4f}{hc/h:>11.1%}   (조건: suc만)")

    # ── ③ 셀별 분포 ─────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("③ (suc, mid, rev) 셀별 결과 구성 — 어디서 갈리는가")
    print("=" * 92)
    d = pd.DataFrame({"suc": suc, "mid": mid, "rev": rev,
                      "ball": ball, "strike": stk})
    d["inplay"] = ((ball == 0) & (stk == 0)).astype(int)
    g = d.groupby(["suc", "mid", "rev"], observed=True)[["ball", "strike", "inplay"]].mean()
    g["n"] = d.groupby(["suc", "mid", "rev"], observed=True).size()
    g["비중"] = g["n"] / n
    print(g.round(4).to_string())

    # ── ④ 카운트 반응이 mid/rev와 다른가 ────────────────────────────────
    print("\n" + "=" * 92)
    print("④ 카운트 반응 — ball/strike가 mid/rev와 **다른 방향**으로 움직이는가")
    print("=" * 92)
    b = df.loc[ok, "balls_before"].to_numpy().clip(0, 3)
    s = df.loc[ok, "strikes_before"].to_numpy().clip(0, 2)
    d2 = pd.DataFrame({"cnt": [f"{x}-{y}" for x, y in zip(b, s)],
                       "suc": suc, "mid": mid, "rev": rev,
                       "off": np.clip(off, 0, 1), "ball": ball, "strike": stk})
    gg = d2.groupby("cnt", observed=True)[["suc", "mid", "rev", "off", "ball", "strike"]].mean()
    gg["n"] = d2.groupby("cnt", observed=True).size()
    print(gg.round(4).to_string())

    # 카운트 프로파일 사이의 상관 — 축이 겹치면 1에 가깝다
    print("\n  카운트 프로파일(12개 셀) 사이 상관계수:")
    prof = gg[["mid", "rev", "off", "ball", "strike"]]
    print(prof.corr().round(3).to_string())
    print("\n  ← ball/strike가 mid·rev·off 중 하나와 |r|≈1 이면 같은 축을 다시 보는 것이다.")


if __name__ == "__main__":
    main()
