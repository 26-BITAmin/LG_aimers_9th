"""§7.118-E — (soft label ON, aux OFF) 미측정 칸: 보조 헤드는 현 스택에서도 밥값을 하나.

왜 지금 다시 재나
    aux(λ=1.0, mr) 채택(§7.41, LB +2.66)은 soft label(§7.70)·init_score(s29)
    **이전** 스택의 측정이다. soft label 은 복제 F 행의 보조 라벨을 강제하며 그
    자체가 라벨 연화 정규화 — §7.42 가 잰 대로 aux 이득의 상당분이 정규화
    성분(rand 가 mode 의 81%)이므로 **부분 대체재일 수 있다** (soft×init 이
    대체재로 판명된 §7.109 와 같은 패턴). N_ROUNDS(§7.112)·half_life(§7.113)는
    신스택 재검을 받았는데 aux ON/OFF 만 안 받았다.

팔 (LGBM z 는 세 팔 공통 — rounds 캐시 k320 = s29 상태)
    A  배포        MLP z = 배포 캐시 (aux ON, 구코드)
    H  하네스      mlpinit_fold*_off.npz 재사용 (§7.114 의 offset=None 팔
                   = aux ON, 같은 신코드) — A 를 재현해야 한다 (§7.114 에서 검증됨)
    C  aux OFF     y_aux=None, 헤드 없음 — 이번에 학습하는 유일한 팔 (8멤버×2fold)

판정 (사전 등록)
    기대 결과는 **현행 유지 확인**(H ≥ C): 재검 전례 2/2 가 현행 유지였다.
    행동 트리거는 C 가 H 를 양 fold 에서 유의하게 이길 때뿐 — 그땐 aux 가
    soft 스택에서 순정규화 비용이 됐다는 뜻이라 제거를 논의한다 (§7.111 기준).
    ⚠️ 재학습 축 — 시드 노이즈(std 4~9)가 낀다. 부트스트랩은 표집 오차만 담는다.

실행 (GPU, 피어 세션 학습 종료 후):
    MLP_DEVICE=cuda .venv-gpu/Scripts/python.exe -u src/mlpaux.py 2>&1 | tee experiments/mlpaux.log
    (GPU venv 부재 시 CPU: .venv/Scripts/python.exe — 시간만 늘어난다)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache as _fc

_fc.CENTER_ASOF = False
KEY = _fc._key()

import pandas as pd                                                 # noqa: E402
import mlp                                                          # noqa: E402
import soften                                                       # noqa: E402
import jointbeta as J                                               # noqa: E402
from config import (ASOF_AXES, CONTAM_AXES, CTX_AXES, DROP_FEATURES,  # noqa: E402
                    MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS,
                    RECENCY_HALF_LIFE, SMOOTH_K_BATTER, SMOOTH_K_PITCHER,
                    TARGET_COL, TRAIN_CSV, USE_PLAYER_IDS)
from data import load_csv, recency_weight                           # noqa: E402
from features import build_features, fit_artifacts                  # noqa: E402
from validation import bss, shift_to_prior                          # noqa: E402

FOLDS = [2023, 2024]
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEED = 42


def train_noaux(df, fold, t0):
    """aux 헤드 없는 8멤버 풀. mlpinit.train_mlps 동형 — y_aux/offset 만 None."""
    pth = CACHE / f"mlpaux_fold{fold}_noaux.npz"
    if pth.exists():
        print(f"    캐시 적중 {pth.name}")
        return np.load(pth)["z"]

    s = df["season"].to_numpy()
    tr, va = df[s < fold], df[s == fold]
    y_tr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr.season.max()))
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)

    def X(d):
        x = build_features(d, art)
        if USE_PLAYER_IDS:
            for c in ("pitcher_id", "batter_id"):
                x[c] = d[c].to_numpy(dtype=np.float32)
        return x.drop(columns=[c for c in DROP_FEATURES if c in x.columns])

    Xtr, Xva = X(tr), X(va)
    # soft label 확장은 유지 (배포와 동일) — aux 라벨만 안 만든다
    Xtr, y2, w2, _, _ = soften.soften(tr, Xtr, y_tr, w, None)

    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    zs = []
    for vname, (cfg, nseed) in MLP_VARIANTS.items():
        for sd in MLP_SEEDS[:nseed]:
            _, curve = mlp.train_mlp(A, C, y2, w2, B, Cv, sizes, seed=sd,
                                     verbose=False, y_aux=None, offset=None,
                                     **dict(cfg, aux=None))
            zs.append(curve[-1])
            print(f"      {vname}/{sd} ({time.time()-t0:.0f}s)")
    z = np.mean(zs, axis=0)
    np.savez_compressed(pth, z=z)
    return z


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"].to_numpy())
    n_id = len(uniq)
    mclean = ~((gt == "F") & (s < 2023))
    dep = np.array([b for _, b in CTX_AXES] + [b for _, b in ASOF_AXES])
    print(f"train {df.shape}   α={MLP_ALPHA}   device={mlp._device()}\n")

    R, KEEP, RAW = {}, {}, {}
    for fold in FOLDS:
        Xax, _, _, _ = J.axis_matrix(df, s, fold)
        vi = np.where(s == fold)[0]
        mm = mclean[vi]
        yv = y[vi]
        with np.load(CACHE / f"rounds_fold{fold}_r640.npz") as d:
            zl = d["k320"]
        with np.load(CACHE / f"fold{fold}_{KEY}.npz") as d:
            zm_dep = d["z_mlp"]
            assert np.allclose(d["y"], yv)
        h_path = CACHE / f"mlpinit_fold{fold}_off.npz"
        arms = {"A 배포(aux ON)": zm_dep,
                "H 하네스(aux ON 신코드)": np.load(h_path)["z"],
                "C aux OFF": train_noaux(df, fold, t0)}
        for lab, zm in arms.items():
            RAW[(fold, lab)] = zm
            z = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm + Xax @ dep
            p = shift_to_prior(1 / (1 + np.exp(-z[mm])), yv[mm].mean())
            R[(fold, lab)] = bss(yv[mm], p)
            KEEP[(fold, lab)] = (yv[mm], z[mm], codes[vi][mm])
        print(f"  fold {fold}: " + "  ".join(f"{k} {R[(fold,k)]:.1f}" for k in arms))

    labs = ["A 배포(aux ON)", "H 하네스(aux ON 신코드)", "C aux OFF"]
    print(f"\n{'='*92}\n█ 배포 층 (blend α={MLP_ALPHA} + δ축 4 + prior shift, clean 행)\n{'='*92}")
    print(f"  {'팔':<26}" + "".join(f"{f'fold {f}':>11}" for f in FOLDS))
    for lab in labs:
        print(f"  {lab:<26}" + "".join(f"{R[(f,lab)]:>11.1f}" for f in FOLDS))

    print(f"\n  ★ 하네스 — H 가 A 를 재현하는가 (§7.114 에서 검증된 팔의 재사용)")
    ok = True
    for f in FOLDS:
        d = R[(f, "H 하네스(aux ON 신코드)")] - R[(f, "A 배포(aux ON)")]
        c = float(np.corrcoef(RAW[(f, "H 하네스(aux ON 신코드)")],
                              RAW[(f, "A 배포(aux ON)")])[0, 1])
        good = abs(d) < 3.0 and c > 0.999
        ok &= good
        print(f"     fold {f}: ΔBSS {d:+.2f}   로짓 상관 {c:.6f}   {'✅' if good else '❌'}")
    print(f"     → {'통과. C 를 읽는다.' if ok else '실패 — C 를 읽지 말 것.'}")

    print(f"\n  클러스터 부트스트랩 (투수 복원추출 200회) — **H 대비** C (aux 제거 효과)")
    for f in FOLDS:
        yv, z1, cf = KEEP[(f, "C aux OFF")]
        z0 = KEEP[(f, "H 하네스(aux ON 신코드)")][1]
        order = np.argsort(cf, kind="stable")
        bnd = np.searchsorted(cf[order], np.arange(n_id + 1))
        seen = np.where(np.diff(bnd) > 0)[0]
        bs = []
        for _ in range(200):
            idx = np.concatenate([order[bnd[j]:bnd[j + 1]] for j in
                                  rng.choice(seen, size=len(seen), replace=True)])
            yy = yv[idx]
            bs.append(bss(yy, shift_to_prior(1 / (1 + np.exp(-z1[idx])), yy.mean()))
                      - bss(yy, shift_to_prior(1 / (1 + np.exp(-z0[idx])), yy.mean())))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        d = R[(f, "C aux OFF")] - R[(f, "H 하네스(aux ON 신코드)")]
        print(f"     fold {f}: aux 제거 Δ {d:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]")

    print(f"\n{'='*92}\n■ 판정\n{'='*92}")
    print("  기대 결과 = 현행 유지 (Δ ≤ 0 또는 구간이 0 포함): aux 는 soft 스택에서도 유효.")
    print("  행동 트리거 = C 가 양 fold 유의 양수일 때만 (§7.111 기준) — aux 제거 논의.")
    print("  ⚠️ 시드 노이즈 std 4~9 — 부트스트랩 구간은 표집 오차만 담는다.")


if __name__ == "__main__":
    main()
