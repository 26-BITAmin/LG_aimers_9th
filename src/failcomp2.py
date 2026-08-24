"""failcomp 후속 — 후보 B 를 **올바른 부분(part)** 으로 다시 짠다.

failcomp P2 가 문서의 전제를 깼다:
    success + middle + reverse + ball = 1.2645   ← ball 은 중복 계수
    ball + strike                     = 0.8141   ← 이 둘도 1 이 아니다
    (middle, ball, reverse) 는 단체(simplex) 가 아니다 → CLR 은 정당한 좌표계가 아니다.

그런데 같은 표에 답이 있다:
    success + middle + reverse = 0.8923 ± 0.0363
    → 부족분 0.1077 이 **컬럼 없는 실패 모드 ②(존에서 크게 벗어남)** 다.
      대회 정의가 실패 3모드이고 우리는 ①middle ③reverse 만 컬럼으로 갖고 있다.

즉 문서의 아이디어(조성으로 수준을 나눈다)는 맞고 **세 번째 부분을 잘못 골랐다.**
올바른 단체는 (middle, reverse, wild) 이고 wild = 1 − success − middle − reverse 다.
이건 기존 컬럼의 **선형결합이라 트리가 축 정렬로는 못 만든다** — 정확히 문서가 말한 자리.

측정 (학습 0회, 배포 잔차 CR0):
    wild        암묵 모드② 비율 그 자체
    wild_share  wild / (middle + reverse + wild)   실패 대비 비율
    clr3_*      (middle, reverse, wild) 의 CLR 좌표 3개
  눈금: fail_middle_share(배포중) ≈ 0, 요일(죽은 축) = 후보와 같은 프로파일이면 사망.

실행:  .venv/Scripts/python.exe -u src/failcomp2.py 2>&1 | tee experiments/failcomp2.log
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
from failcomp import clr
from tmslot import resid_t_cr0


def main():
    df, s, y = D.load_df()
    for fold in D.FOLDS:
        st = D.state(df, s, y, fold)
        d = st["d"]
        get = lambda c: d[f"asof_pitcher_{c}_rate"].to_numpy(np.float64)
        m, r_, su, b, stk = (get(c) for c in ("middle", "reverse", "success", "ball", "strike"))
        for v in (m, r_, su, b, stk):
            np.copyto(v, np.nanmean(v), where=np.isnan(v))

        wild = 1.0 - su - m - r_
        fail = m + r_ + np.clip(wild, 0, None)
        C = clr(np.stack([m, r_, np.clip(wild, 1e-6, None)], axis=1))

        base = np.column_stack([m, r_, b, su, stk, m / np.maximum(m + r_, 1e-9),
                                b - stk, np.ones(len(m))])
        rows = {
            "wild (암묵 모드②)":  wild,
            "wild_share":          np.clip(wild, 0, None) / np.maximum(fail, 1e-9),
            "clr3_middle":         C[:, 0],
            "clr3_reverse":        C[:, 1],
            "clr3_wild":           C[:, 2],
            "── fail_middle_share (배포중)": m / np.maximum(m + r_, 1e-9),
            "── 요일 (죽은 축)":   d["game_dayofweek"].to_numpy(np.float64),
        }
        print(f"\nfold {fold}   clean {len(m):,}행   기준 BSS {st['base']:.1f}"
              f"   wild 평균 {wild.mean():.4f} ± {wild.std():.4f}"
              f"   음수 {100*(wild<0).mean():.2f}%")
        print(f"  {'좌표':<30s}{'기존 선형 R²':>14s}{'잔차 CR0 t':>13s}{'기울기·1sd':>13s}")
        for nm, v in rows.items():
            coef, *_ = np.linalg.lstsq(base, v, rcond=None)
            r2 = 1.0 - (v - base @ coef).var() / max(v.var(), 1e-12)
            vz = (v - v.mean()) / max(v.std(), 1e-12)
            sl, t = resid_t_cr0(st["r"], vz, st["cl"], st["n_id"])
            tag = "  <<<" if (abs(t) > 2 and not nm.startswith("──")) else ""
            print(f"  {nm:<30s}{r2:>14.4f}{t:>+13.2f}{sl:>+13.5f}{tag}")
    print("\n주: wild 는 success·middle·reverse 의 선형결합이라 선형 R²=1 이 정상이다.")
    print("    트리는 축 정렬이라 이 결합을 못 만든다 — 그래서 R² 가 아니라 **잔차 t** 가 판정자다.")


if __name__ == "__main__":
    main()
