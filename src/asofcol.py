"""§7.56 — 시즌 진행분을 **컬럼으로** 넣으면 사후 가산보다 나은가.

왜 묻는가
    §7.55는 사후 로짓 가산으로 배포했고 LB +26.14를 받았다. 그 형태는 **선형**이다:
    z' = z + β·gap.  모델은 gap을 카운트·손잡이·주자상황과 **곱해서** 쓸 수 없다.

    도메인적으로 상호작용이 있을 자리가 있다. 예를 들어 "올해 제구가 무너진 투수"의
    피해는 3볼(볼넷 회피로 존 가운데)에서 2스트라이크(유인구)보다 클 수 있다.
    §7.37이 카운트 축을 δ로 건진 것과 같은 종류의 구조다.

    컬럼으로 넣으면 트리가 그 상호작용을 만들 수 있다. 대신 §7.51이 본
    **'컬럼 추가 비용'** 을 낸다 — 분할 후보가 늘고 feature_fraction과 상호작용해
    정보와 무관하게 점수가 움직인다.

⚠️ 이 하네스는 **LGBM 단독**이다 (블렌드·δ·오염보정 없음). §7.51·§7.15가 쓴 그 하네스고,
   fold2023 기준선이 −11.1인 것도 그래서다. 배포 시스템의 절대값이 아니라
   **전달 방식 사이의 차이**를 보려고 쓴다.

변형
    base        아무것도 없음
    post        사후 가산 β=0.02          ← 배포 형태
    col1        gap_success 컬럼 1개
    col4        gap_success + r_szn + n_szn + gap_middle
    col1_post   컬럼 + 사후 가산          ← 상호작용이 있으면 post 대비 이득이 남는다
    col1_shuf   ★ 대조군 — 시즌 안에서 pitcher_id 무작위 재배정

판정
    핵심 대비는 **col1_post − post** 다. 양수여야 "선형으로 다 못 짜냈다"가 된다.
    col1 − base 가 크더라도 post − base 와 비슷하면 컬럼으로 바꿀 이유가 없다
    (재학습 비용 + 컬럼 추가 비용을 내고 같은 것을 얻는 셈).

실행:  .venv/Scripts/python.exe -u src/asofcol.py 2>&1 | tee experiments/asofcol.log
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
from config import ASOF_BETA, ASOF_STRENGTH, SEED, TARGET_COL, TRAIN_CSV
from data import load_csv
from evaluate import compare

COLS4 = ["gap_success", "r_szn_success", "n_szn", "gap_middle"]


def _state(tr, shuffle=False):
    """학습 시즌으로 오프셋·표준화 상수를 fit 한다. **검증 행은 안 본다.**"""
    a = asofdelta.fit_asof(tr, 1.0, ASOF_STRENGTH)          # β는 여기서 안 쓴다
    off = asofdelta.fit_offsets(tr)
    league = float(tr[TARGET_COL].mean())
    # 나머지 컬럼의 표준화 통계도 학습 행에서만
    raw_tr = _raw(tr, off, league)
    mu = {c: float(np.nanmean(v)) for c, v in raw_tr.items()}
    sd = {c: float(np.nanstd(v)) or 1.0 for c, v in raw_tr.items()}
    return {"a": a, "off": off, "league": league, "mu": mu, "sd": sd,
            "shuffle": shuffle}


def _raw(d, off, league):
    """원단위 시즌 진행분 피처. 행 단위 조회뿐."""
    st = asofdelta.season_to_date(d, off)
    n = st["asof_pitcher_n"]
    r_car = d["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    prior = np.where(np.isfinite(r_car), r_car, league)
    r_szn = asofdelta.shrink(st["asof_pitcher_success_rate"], n, r_car, league)

    r_mid = d["asof_pitcher_middle_rate"].to_numpy(dtype=np.float64)
    pm = np.where(np.isfinite(r_mid), r_mid, 0.2)
    return {"gap_success": r_szn - prior,
            "r_szn_success": r_szn,
            "n_szn": n,
            "gap_middle": asofdelta.shrink(st["asof_pitcher_middle_rate"], n,
                                           r_mid, 0.2) - pm}


def make_extra(cols, shuffle=False):
    """compare()에 넘길 컬럼 훅. `extra(tr) -> f(d) -> {컬럼: 배열}`"""
    def fit(tr):
        s = _state(tr, shuffle)

        def apply(d):
            dd = d
            if shuffle:                    # ★ 대조군 — 시즌 안에서 투수만 섞는다
                dd = d.copy()
                rng = np.random.default_rng(SEED)
                dd["pitcher_id"] = rng.permutation(d["pitcher_id"].to_numpy())
            raw = _raw(dd, s["off"], s["league"])
            return {c: np.nan_to_num((raw[c] - s["mu"][c]) / s["sd"][c])
                    for c in cols}
        return apply
    return fit


def make_post(beta=ASOF_BETA):
    """사후 가산 훅 — 배포와 **같은 코드**(`features.asof_lookup`)를 쓴다."""
    def fit(tr):
        a = asofdelta.fit_asof(tr, beta, ASOF_STRENGTH)
        return lambda d: float(a["beta"]) * F.asof_lookup(d, a)
    return fit


def main():
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   β(배포)={ASOF_BETA}  strength={ASOF_STRENGTH}")
    print("⚠️ LGBM 단독 하네스 — 절대값이 아니라 **전달 방식 간 차이**를 본다\n")
    compare(df, {
        "base":      dict(),
        "post":      dict(post=make_post()),                       # 배포 형태
        "col1":      dict(extra=make_extra(["gap_success"])),
        "col4":      dict(extra=make_extra(COLS4)),
        "col1_post": dict(extra=make_extra(["gap_success"]), post=make_post()),
        "col1_shuf": dict(extra=make_extra(["gap_success"], shuffle=True)),  # 대조군
    })
    print("\n" + "=" * 78)
    print("읽는 법 — 핵심 대비는 **col1_post − post**")
    print("=" * 78)
    print("  양수  → 선형 사후 가산으로 다 못 짜냈다. 상호작용이 남아 있다")
    print("  0 근처 → 컬럼으로 바꿀 이유가 없다 (재학습 비용만 내고 같은 것을 얻는다)")
    print("  col1 − base 가 post − base 와 비슷하면 그것도 같은 결론이다.")


def decomp():
    """col4 − col1 = +39.2 는 어느 컬럼에서 오나.

    가설: **`n_szn`이 신뢰도를 알려준다.** 사후 가산은 30구짜리 추정치와 2000구짜리
    추정치에 같은 β를 곱한다 (선형이라 구조적으로 그럴 수밖에 없다). 컬럼이면
    트리가 "n_szn이 클 때만 gap을 믿어라"라는 **상호작용**을 만들 수 있다.
    맞다면 `col1n`이 `col4`의 대부분을 설명해야 한다.

    ⚠️ 대조군도 컬럼 수를 맞춘다 — `col4_shuf`는 4개짜리다. 1개짜리 대조군으로
       4개짜리 변형을 판정하면 '컬럼 추가 비용'을 과소평가한다 (§7.51).
    """
    df = load_csv(TRAIN_CSV)
    print(f"train {df.shape}   col4 분해\n")
    compare(df, {
        "base":      dict(),
        "post":      dict(post=make_post()),                        # 참조 (배포 형태)
        "col1":      dict(extra=make_extra(["gap_success"])),
        "col1n":     dict(extra=make_extra(["gap_success", "n_szn"])),       # ★ 가설
        "col1r":     dict(extra=make_extra(["gap_success", "r_szn_success"])),
        "col4":      dict(extra=make_extra(COLS4)),
        "col4_shuf": dict(extra=make_extra(COLS4, shuffle=True)),   # ★ 4개짜리 대조군
    })


if __name__ == "__main__":
    decomp() if "--decomp" in sys.argv else main()
