"""MLP 멤버의 입력 부분집합 — 남은 유일한 축

동기
    `noid`(투수/타자 임베딩 제거)가 이 수의 유일한 사례이고 **채택됐다**
    (`ablate17` ②: 단독 +27.0, emb4와 묶어 +24.2로 배포). 같은 수를 **다른 블록으로는
    한 번도 안 해봤다.** §7.11의 카테고리 표가 후보를 이미 알려준다 (제거 시 시스템 손해):

        손잡이 -96.9  |  직전경기 -40.5  |  팀ID -21.9  |  asof원본 -21.2  |  선수ID -9.5

    큰 숫자 = 그 블록이 시스템에 필수 = **빼면 그 멤버는 크게 다른 함수가 된다.**
    MLP는 핸디캡 -223짜리도 corr 0.83으로 이겼다 (§7.13). 기준은 핸디캡이 아니라
    **핸디캡 대비 탈상관 보너스의 비율**이다.

    ⚠️ 이건 §7.15의 "정보 삭감"(LB -2.13)과 다르다. 거기서는 **시스템 전체**에서
    피처를 뺐다. 여기서는 LGBM 12부스터와 기존 MLP 8멤버가 전부 그대로 있고,
    **한 멤버만 눈을 가린 채 추가**한다. 시스템의 정보량은 줄지 않는다. noid가 그 증거다.

설계
    · cfg는 `emb4`와 **동일**하게 두고 입력 블록만 뺀다 (자유 파라미터를 늘리지 않는다).
      에폭도 9로 고정 — valid argmax 금지 (§7.13의 정직한 재판정 규약).
    · 신규 멤버는 기존 8멤버 풀에 **추가**된다: pool = (8·z_mlp + Σ신규) / (8+k).
      멤버별 z가 없어도 평균만으로 계산된다.
    · LGBM·기존 MLP는 foldcache에서 그대로 읽는다 (재학습 0).
    · α는 배포값 0.40 고정. α까지 같이 움직이면 귀속이 흐려진다 (§7.22에서 α는 닫혔다).

사전 예측 (실행 전에 박아둔다)
    P1  corr(신규멤버, 현행블렌드)는 nohand < noform < noraw 순.   내 예측: **그렇다**
        (뺀 블록이 클수록 다른 함수가 된다)
    P2  noraw는 기준 대비 |Δ| < 3.  내 예측: **그렇다**
        (원본 asof는 `_sm` 버전과 거의 중복 — §7.11에서 스무딩 제거가 +11.0이었다)
    P3  조건 ①②③을 통과하는 후보 수.  내 예측: **0개**
        (이 프로젝트의 기저율이 그렇다. 그래도 P1이 맞으면 다음 후보를 고를 축이 생긴다)

실행:  .venv/Scripts/python.exe -u src/ablate25.py 2>&1 | tee experiments/ablate_run25.log
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import ablate23 as A
import foldcache
import mlp
from config import (MLP_ALPHA, MLP_SEEDS, MLP_VARIANTS, RECENCY_HALF_LIFE,
                    SMOOTH_K_BATTER, SMOOTH_K_PITCHER, TARGET_COL, TRAIN_CSV)
from data import load_csv, recency_weight
from features import fit_artifacts
from foldcache import CACHE, _key
from mlp_infer import encode_apply

THREADS = 5
mlp.THREADS = THREADS

FOLDS = [2023, 2024]
SEEDS = MLP_SEEDS[:3]
N_OLD = sum(k for _, k in MLP_VARIANTS.values())        # 현행 풀 크기 (8)

# §7.11의 카테고리. 숫자는 "시스템에서 제거했을 때의 손해" — 클수록 필수 블록이다.
HAND = ["pitcher_hand", "batter_hand", "hand_same"]                          # -96.9
FORM = ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "form_p1_minus_career", "form_p3_minus_career", "form_p1_minus_p5",
        "form_p3_minus_p5", "form_mid_p1_minus_career"]                      # -40.5
RAW = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
       "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_batter_success_rate",
       "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
       "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]           # -21.2

EMB4 = MLP_VARIANTS["emb4"][0]                          # cfg를 그대로 물려받는다
NEW = {                                                 # 이름: (뺄 숫자컬럼, 뺄 임베딩)
    "nohand": (HAND, ("hand_matchup",)),
    "noform": (FORM, ()),
    "noraw":  (RAW, ()),
}


def ck(fold, name, seed):
    return CACHE / f"sub_{name}_f{fold}_{_key()}_{seed}.npz"


def fold_inputs(df, fold):
    """MLP 학습에 필요한 것만. LGBM은 캐시에서 읽으므로 여기서 안 만든다."""
    season = df.season.to_numpy()
    tr, va = df[season < fold], df[season == fold]
    w = recency_weight(tr.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=fold - 1)
    art = fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                        k_batter=SMOOTH_K_BATTER)
    Xtr, Xva = A.featurize(tr, art), A.featurize(va, art)
    ytr = tr[TARGET_COL].to_numpy(dtype=np.float64)
    enc = mlp.fit_encoder(Xtr, art)
    An, Ctr = encode_apply(Xtr, enc)
    B, Cva = encode_apply(Xva, enc)
    del tr, va, Xtr, Xva
    gc.collect()
    return An, Ctr, ytr, w, B, Cva, enc


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = foldcache.load(FOLDS)
    Z = {f: (cached[f][0], cached[f][1], cached[f][2]) for f in FOLDS}

    need = [(f, n, s) for f in FOLDS for n in NEW for s in SEEDS
            if not ck(f, n, s).exists()]
    df = load_csv(TRAIN_CSV) if need else None

    new_z = {}
    for fold in FOLDS:
        inputs = None
        for name, (dnum, demb) in NEW.items():
            for s in SEEDS:
                p = ck(fold, name, s)
                if p.exists():
                    new_z[(fold, name, s)] = np.load(p)["z"]
                    continue
                if inputs is None:
                    inputs = fold_inputs(df, fold)
                    print(f"\n=== fold {fold} 인코딩 완료 ({time.time()-t0:.0f}s) ===")
                An, Ctr, ytr, w, B, Cva, enc = inputs
                # 숫자 컬럼은 여기서 마스킹한다. 표준화는 컬럼별로 독립이라
                # mu/sd를 다시 구할 필요가 없다.
                keep = [i for i, c in enumerate(enc["num_cols"]) if c not in dnum]
                cfg = dict(EMB4, drop_emb=demb)
                m, _ = mlp.train_mlp(An[:, keep], Ctr, ytr, w, An[:512, keep],
                                     Ctr[:512], enc["sizes"], seed=s,
                                     verbose=False, **cfg)
                z = mlp.infer(m, B[:, keep], Cva)
                np.savez_compressed(p, z=z)
                new_z[(fold, name, s)] = z
                print(f"  {name}_{s}  숫자 {len(keep)}/{len(enc['num_cols'])}개"
                      f"  단독 {mlp.score(z, Z[fold][2]):>8.1f}  ({time.time()-t0:.0f}s)")
                del m
        del inputs
        gc.collect()
    del df
    gc.collect()

    def bl(zl, zm, y):
        return mlp.score((1 - MLP_ALPHA) * zl + MLP_ALPHA * zm, y)

    base = {f: bl(*Z[f]) for f in FOLDS}
    print("\n" + "=" * 100)
    print(f"신규 멤버 진단 (3시드 평균, 기준 = 현행 {N_OLD}멤버 풀, α={MLP_ALPHA})")
    print("=" * 100)
    print(f"{'변형':>10} {'fold':>6} {'단독':>9} {'핸디캡':>9} {'corr(현행블렌드)':>18}")
    for name in NEW:
        for f in FOLDS:
            zl, zm, y = Z[f]
            zn = np.mean([new_z[(f, name, s)] for s in SEEDS], axis=0)
            cur = (1 - MLP_ALPHA) * zl + MLP_ALPHA * zm
            print(f"{name:>10} {f:>6} {mlp.score(zn, y):>9.1f}"
                  f" {mlp.score(zn, y) - mlp.score(zm, y):>9.1f}"
                  f" {np.corrcoef(zn, cur)[0, 1]:>18.3f}")

    print("\n" + "=" * 100)
    print(f"채택 판정 — 신규 멤버를 풀에 추가.  pool = ({N_OLD}·z_mlp + Σ신규) / ({N_OLD}+k)")
    print("=" * 100)
    print(f"  기준  " + "  ".join(f"fold{f} {base[f]:.1f}" for f in FOLDS))

    combos = {n: [n] for n in NEW}
    combos["nohand+noform"] = ["nohand", "noform"]
    combos["셋 전부"] = list(NEW)

    for label, names in combos.items():
        d = {}
        for f in FOLDS:
            zl, zm, y = Z[f]
            add = [new_z[(f, n, s)] for n in names for s in SEEDS]
            pool = (N_OLD * zm + np.sum(add, axis=0)) / (N_OLD + len(add))
            d[f] = bl(zl, pool, y) - base[f]
        sign = len({np.sign(v) for v in d.values()}) == 1
        mags = sorted(abs(v) for v in d.values())
        ratio = mags[-1] / mags[0] if mags[0] > 1e-9 else float("inf")
        avg = float(np.mean(list(d.values())))
        ok = sign and avg > 0 and ratio <= 3
        print(f"  {label:>14}  " + "  ".join(f"Δfold{f} {d[f]:+8.2f}" for f in FOLDS)
              + f"   평균 {avg:+8.2f}   부호 {'O' if sign else 'X'}"
              + f"   크기비 {ratio:>6.1f}   → {'채택 후보' if ok else '기각'}")

    print("\n  ① 효과>노이즈  ② 부호일치  ③ 크기비≤3  — 조건 ③은 4전 4중이다. 넘기지 않는다.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
