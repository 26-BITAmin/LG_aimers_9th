""":§7.106 본 실험 — `CENTER_ASOF` 켜고 fold 재학습해서 배포 스택과 정면 비교.

§7.105 까지의 모든 조작은 **얼린 z 위의 사후 변환**이라 재학습이 필요 없었다.
이번은 **피처 자체가 바뀌므로** 재학습 말고는 판정 방법이 없다 —
중심화는 한 시즌 안에서는 상수 이동이라 추론 시점엔 no-op 이고, 효과는 전부
"풀링된 여러 시즌에서 무엇을 배우는가"에 있다.

비교 층 (둘 다 δ축은 동일 — axis_matrix 는 원본 컬럼에서 만들어지므로 팔에 무관)
    blend    (1−α)·z_lgb + α·z_mlp
    deploy   blend + δ_hand + δ_count4 + gap_success + gap_bsuccess

⚠️ fold 의 검증 시즌은 중심화 표에 **없다** → 외삽 경로를 탄다. 2025 와 같은 조건이다.
   (실측 외삽 오차: 2024 예측 0.51337 vs 실제 0.51140 = 컬럼 std 의 0.04배)

실행:  .venv/Scripts/python.exe -u src/asofctr2.py 2>&1 | tee experiments/asofctr2.log
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import config

FOLDS = [2023, 2024]

# ⚠️⚠️ **지문은 `jointbeta` 를 import 하기 전에 계산해야 한다.**
#   jointbeta.py:60 이 `foldcache._key = lambda: _PIN` 으로 지문을 "5488ef1b18" 에
#   **못 박는다.** config 가 바뀌어도 fold 재학습이 안 시작되게 하는 안전장치인데,
#   이번처럼 **의도적으로 지문을 갈라야 할 때는 정반대로 작동한다** — 두 팔이 같은
#   캐시를 읽어 Δ가 전부 +0.0 으로 나왔다(한 번 속았다).
#   여기서는 pin 이 걸리기 전에 진짜 `_key()` 를 두 번 불러 둔다.
import foldcache as _fc

_KEYS = {}
for _flag in (False, True):
    _fc.CENTER_ASOF = _flag
    _KEYS[_flag] = _fc._key()
_fc.CENTER_ASOF = config.CENTER_ASOF
assert _KEYS[False] != _KEYS[True], "지문이 안 갈렸다 — pin 이 이미 걸렸는지 확인"


def main():
    import foldcache
    import jointbeta as J
    from config import ASOF_AXES, CTX_AXES, MLP_ALPHA, TARGET_COL, TRAIN_CSV
    from data import load_csv
    from validation import bss, shift_to_prior
    import mlp

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])

    # δ축은 팔에 무관하다 — 한 번만 만들어 재사용
    AX = {}
    for f in FOLDS:
        X, _, _, _ = J.axis_matrix(df, s, f)
        AX[f] = X

    res = {}
    for arm, flag in (("raw (배포)", False), ("centered", True)):
        # ⚠️ 모듈 속성만 바꿔서 `_key()`가 갈리길 기대했다가 사고를 냈다 — 두 팔이
        #    같은 캐시를 읽어 Δ가 전부 +0.0으로 나왔다. 지금은 **지문으로 npz를 직접
        #    연다.** 학습은 별도 프로세스에서:
        #      CENTER_ASOF=1 .venv/Scripts/python.exe src/foldcache.py 2023 2024
        key = _KEYS[flag]           # pin 걸리기 전에 계산해 둔 진짜 지문
        print(f"\n{'='*72}\n■ {arm}   지문 {key}\n{'='*72}")
        cached = {}
        for f in FOLDS:
            pth = foldcache.CACHE / f"fold{f}_{key}.npz"
            if not pth.exists():
                raise SystemExit(f"캐시 없음 {pth.name} → CENTER_ASOF={int(flag)} 로 먼저 학습")
            with np.load(pth) as d:
                cached[f] = (d["z_lgb"], d["z_mlp"], d["y"], None)
            print(f"  {pth.name}")
        res[arm] = {}
        for f in FOLDS:
            zl, zm, yv = cached[f][0], cached[f][1], cached[f][2]
            vi = np.where(s == f)[0]
            assert len(vi) == len(yv) and np.allclose(y[vi], yv)
            mm = mclean[vi]
            zb = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
            zd = zb + AX[f] @ dep
            out = {}
            for lab, z in (("lgb", zl), ("mlp", zm), ("blend", zb), ("deploy", zd)):
                p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
                out[lab] = bss(yv[mm], p)
            res[arm][f] = out
            print(f"  fold {f}: " + "  ".join(f"{k} {v:.1f}" for k, v in out.items()))

    print(f"\n{'='*72}\n█ 판정 — clean 행 BSS (검증 시즌은 중심화 표에 없음 = 외삽 경로)\n{'='*72}")
    print(f"  {'fold':<8}{'층':<9}{'raw':>10}{'centered':>11}{'Δ':>9}")
    tot = {}
    for f in FOLDS:
        for lab in ("lgb", "mlp", "blend", "deploy"):
            a, b = res["raw (배포)"][f][lab], res["centered"][f][lab]
            tot.setdefault(lab, []).append(b - a)
            print(f"  {f:<8}{lab:<9}{a:>10.1f}{b:>11.1f}{b-a:>+9.1f}")
    print(f"\n  {'평균 Δ':<17}" + "".join(f"{lab}:{np.mean(v):+.1f}  " for lab, v in tot.items()))
    print(f"  {'부호일치':<17}" + "".join(
        f"{lab}:{'O' if all(np.sign(x)==np.sign(v[0]) for x in v) else 'X'}  "
        for lab, v in tot.items()))
    print("\n  채택 조건 (§2): ① 두 fold 부호 일치 ② 크기비가 상식 범위 ③ deploy 층에서 양수")


if __name__ == "__main__":
    main()
