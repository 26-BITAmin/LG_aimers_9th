"""MLP 깊이 × 잔차 **요인설계** — §7.25 의 교락을 뚫는다 (사용자 제안: depth 5/6/7 + resnet).

■ 왜 다시 재는가 — §7.25 는 대각선만 밟았다
`ablate26.py` ARMS:  emb4ctl = depth2·plain / deep3 = depth3·plain / res4 = depth4·**resid**

           resid=off        resid=on
  depth2   emb4ctl (기준)   ⬜ 없음
  depth3   deep3 +0.6/+1.1  ⬜ 없음
  depth4   ⬜ 없음           res4  -3.6/-2.3

`res4` 는 깊이와 잔차를 **동시에** 바꿨다 → -3.6 을 어느 쪽에도 못 돌린다. 여기서 3x2 를
채워 ① 깊이 주효과 ② 잔차 주효과 ③ **상호작용** `(d6r-d6p)-(d2r-d2p)` 을 가른다.
③ 이 이 제안의 핵심이다 — "깊어질수록 잔차가 필요해지는가". §7.164(GRANDE)에서
상호작용이 주효과의 143~306% 였던 전례가 있다.

■ 용량 산술 (블록 = Linear(h,h)+BatchNorm1d(h)+GELU+Dropout, 층당 +16,768)
  depth2  26.6k (1.0x)  |  depth4  60.2k (2.3x) -> res4 -3.6
  depth6  93.7k (3.5x)  |  wide256 86.0k (3.2x) -> -88.2/-74.9
깊이는 폭보다 훨씬 완만한 축이다(2.3x 에서 -3 vs 3.2x 에서 -88) → §7.26 의 "파라미터를
늘리면 진다" 가 깊이 방향으로 자동 전이되지 않는다. 동시에 depth6 은 wide256 의 파라미터를
넘는다. **어느 쪽으로도 외삽할 근거가 없다** (MEMORY: local-slope-times-octaves-is-broken).

■ 하네스 — `mlpreg.py`(§7.176a 배포 등가) 를 그대로 물려받는다
soften(soft label) + 보조 헤드 lambda=MLP_AUX_LAMBDA + foldcache(contam/center_asof) +
LGBM 캐시(`mlpreg_lgb_f{fold}.npz` 재사용) + MLP_ALPHA 블렌드.
⚠️ §7.176a **이후**의 T1 계보 변경(§7.177 라우팅, §7.181/§7.183a 용량 노브, §7.178
   CatBoost 12)은 **안 들어 있다.** 이 스크립트가 재는 것은 "MLP 팔 내부 구조" 이고,
   통과하면 T1 스택 위 재확인이 별도로 필요하다 (MEMORY: audit-inherited-constants).
⚠️ MLP_ALPHA=0.75 -> 이 팔이 배포 블렌드 로짓의 75% 다. 트리 노브보다 레버리지가 3배.

■ 에폭 규약 (사용자 결정: 고정 9 + 가장 깊은 팔만 곡선)
채점은 **전 팔 epochs=9 고정**. `OneCycleLR(total_steps=steps)` 이고 steps 가 epochs 에서
나오므로(mlp.py:523) 15에폭 런의 9번째 에폭은 9에폭 런과 **다른 모델**이다 — 곡선이
공짜가 아니라 점마다 별도 런이다. 그래서 5점(3/6/9/12/15) 대신 **2점 {9,15}** 로 줄여
가장 깊은 팔(d6r)에서만 돌리고, **미수렴 진단에만** 쓴다. 채점에는 절대 안 쓴다
(valid argmax = 반칙 — §7.25 가 온도 실험에서 스스로 지적한 것).

■ 사전 등록
P1 잔차 주효과 |Δ| < 기준팔 시드 SD               (depth2 에서 Res 는 거의 무연산)
P2 깊이 주효과 단조 음수  d2 > d4 > d6
P3 corr(z_arm, z_d2p) > 0.99 — 다양성이 안 생긴다 (§7.25 는 .965 -> .966)
P4 상호작용 (d6r-d6p)-(d2r-d2p) > 0 — 깊을수록 잔차가 도움이 된다
P5 d6r: (ep15 - ep9) < +2·SE 이면 "고정 9 는 미수렴 아님" -> 격자 판정 유효.
   그보다 크면 격자 결과는 **거짓 음성**이고 표를 읽으면 안 된다.

■ 판정 순서
① P5 게이트. 실패하면 나머지 표는 해석하지 않는다.
② 깊이/잔차 주효과 + 상호작용을 쌍체 t 로. 깊이가 다른 팔끼리는 RNG 소비가 달라
   쌍체가 안 붙는다 -> rho 를 같이 찍는다 (MEMORY: pairing-buys-correlation-only-if-...).
③ 통과 팔만 beta 스윕(무학습). beta*=0 이면 후보 아님.
④ 채택되면 배포 꼬리가 있다 — `export_weights` 는 W0/W1/W2 하드코딩 + BN 접기이고
   `mlp_infer.MLPPredictor` 에는 BatchNorm/Res 경로가 아예 없다(mlp.py:599-640).
   둘을 일반화하고 numpy-torch 0.000e+00 대조를 먼저 해야 한다.

실행:  .venv/Scripts/python.exe -u src/mlpdepth.py 2024
       .venv/Scripts/python.exe -u src/mlpdepth.py 2023
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

import lightgbm as lgb
import numpy as np

import foldcache
import mlp
import modelabel
import soften
from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, MLP_ALPHA, MLP_AUX_LAMBDA,
                    MLP_VARIANTS, N_ROUNDS, TRAIN_CSV)
from data import load_csv

mlp.THREADS = 8
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7, 2024, 1234, 31337)
TCRIT = 2.776                                    # t(0.975, df=4)

EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
BASE = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True,
            emb_dim=EMB4)

# ⚠️ 물려받은 상수 감사 — BASE 가 배포 emb4 변형과 정말 같은가.
#    (MEMORY: audit-inherited-constants-not-just-registered-gates)
assert BASE == MLP_VARIANTS["emb4"][0], (
    f"BASE 가 배포 emb4 설정과 다르다\n  BASE={BASE}\n  emb4={MLP_VARIANTS['emb4'][0]}")

# depth=2·resid=False 는 mlp.py 가 "예전 코드와 완전히 같은 Sequential" 이라고 보증한다
# -> d2p 는 배포 emb4 멤버 그 자체다 (정합성 게이트가 구성상 만족된다).
# resid 요인은 **완벽 쌍체**다: Res 는 파라미터를 안 만들고 block(i) 의 모듈 생성 순서가
# 같으므로 같은 시드의 dXp / dXr 은 초기 가중치가 비트 단위로 동일하다.
ARMS = {
    "d2p": dict(depth=2, resid=False),           # 기준 = 배포 대조
    "d2r": dict(depth=2, resid=True),
    "d4p": dict(depth=4, resid=False),           # §7.25 가 못 잰 칸
    "d4r": dict(depth=4, resid=True),            # = res4 재현
    "d6p": dict(depth=6, resid=False),
    "d6r": dict(depth=6, resid=True),
}
GATE_ARM = ("d6r_e15", dict(depth=6, resid=True, epochs=15))   # P5 전용, 채점 제외
DEPTHS = (2, 4, 6)


def get_lgb(X, y, w, Xva, yva, fold, t0):
    """z_lgb 는 모든 팔에서 고정 -> 한 번만 학습해 캐시. mlpreg.py 와 **같은 파일**을 쓴다."""
    f = CACHE / f"mlpreg_lgb_f{fold}.npz"
    if f.exists():
        z = np.load(f)["z"]
        print(f"  [{time.time()-t0:.0f}s] LGBM 재사용  단독 {mlp.score(z, yva):.1f}")
        return z
    z, n = np.zeros(len(Xva)), 0
    for vover in ENSEMBLE_VARIANTS.values():
        for s in (42, 7, 2024):
            p = dict(FINAL_LGB_PARAMS, **vover, seed=s, bagging_seed=s,
                     feature_fraction_seed=s, data_random_seed=s)
            bst = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=N_ROUNDS)
            z += bst.predict(Xva, raw_score=True)
            n += 1
            del bst
    z /= n
    np.savez_compressed(f, z=z)
    print(f"  [{time.time()-t0:.0f}s] LGBM {n}개  단독 {mlp.score(z, yva):.1f}")
    return z


def run_arm(name, cfg, A, C, y, w, B, Cv, sizes, aux, z_lgb, yva, fold, t0):
    """팔 하나를 시드별로 학습. 점수 배열 (n_seed,) 와 시드평균 MLP 로짓을 돌려준다."""
    f = CACHE / f"mlpdepth_f{fold}_{name}.npz"
    if f.exists():
        d = np.load(f)
        print(f"[{time.time()-t0:.0f}s] {name:<8} 재사용 {d['s'].mean():8.2f}")
        return d["s"], d["zm"]
    per_seed, zsum = [], None
    for s in SEEDS:
        _, curve = mlp.train_mlp(A, C, y, w, B, Cv, sizes, seed=s, verbose=False,
                                 y_aux=aux, **dict(BASE, **cfg, aux=MLP_AUX_LAMBDA))
        zm = curve[-1]
        zsum = zm.astype(np.float64) if zsum is None else zsum + zm
        per_seed.append(mlp.score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * zm, yva))
        del curve
        gc.collect()
    sc, zbar = np.array(per_seed), (zsum / len(SEEDS)).astype(np.float32)
    np.savez_compressed(f, s=sc, zm=zbar)
    print(f"[{time.time()-t0:.0f}s] {name:<8} 시드별 {sc.mean():8.2f} ± {sc.std(ddof=1):.2f}"
          f"   ({'  '.join(f'{v:.0f}' for v in sc)})")
    return sc, zbar


def paired(d, label, width=34):
    """쌍체 차분 벡터 -> 평균/SE/t/부호/95%CI 한 줄."""
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0.0
    sign = f"{int((d > 0).sum())}/{len(d)}"
    return (f"{label:<{width}}{d.mean():>9.2f}{se:>8.2f}{t:>8.2f}{sign:>7}"
            f"   [{d.mean()-TCRIT*se:+7.2f}, {d.mean()+TCRIT*se:+7.2f}]")


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    del df
    gc.collect()
    aux0, cov = modelabel.aux_labels(tr)
    X, y, w, aux, h = soften.soften(tr, Xtr0, ytr0, w0, aux0)
    print(f"[{time.time()-t0:.0f}s] fold={fold}  soft {len(Xtr0):,}→{len(X):,}행"
          f"  보조라벨 {cov:.1%}  α={MLP_ALPHA} (MLP 가 블렌드의 {MLP_ALPHA:.0%})")

    z_lgb = get_lgb(X, y, w, Xva, yva, fold, t0)
    A, C, B, Cv, _, sizes = mlp.encode(X, Xva, art)
    del Xtr0, X
    gc.collect()

    S, Z = {}, {}
    for name, cfg in list(ARMS.items()) + [GATE_ARM]:
        S[name], Z[name] = run_arm(name, cfg, A, C, y, w, B, Cv, sizes, aux,
                                   z_lgb, yva, fold, t0)

    base, gname = S["d2p"], GATE_ARM[0]
    print(f"\n{'='*88}\n[fold {fold}] 기준팔 d2p {base.mean():.2f} "
          f"(시드 SD {base.std(ddof=1):.2f}, n={len(SEEDS)})\n{'='*88}")

    # ── ① P5 게이트 — 고정 9 가 depth6 에서 미수렴인가 ──────────────────────
    dg = S[gname] - S["d6r"]
    seg = dg.std(ddof=1) / np.sqrt(len(dg))
    print("\n[① P5 에폭 게이트]  d6r: ep15 − ep9   (미수렴 진단 전용, 채점 제외)")
    print("  " + paired(dg, "ep15 − ep9", 30))
    ok5 = dg.mean() < 2 * seg
    print(f"  → {'통과' if ok5 else '실패'} — "
          + ("고정 9 는 미수렴이 아니다. 아래 표 유효."
             if ok5 else "9에폭이 부족하다. **아래 표는 거짓 음성이므로 읽지 말 것.**"))

    # ── ② 팔별 Δ ─────────────────────────────────────────────────────────
    print(f"\n[② 팔별 Δ vs d2p]\n{'팔':<34}{'쌍체Δ':>9}{'SE':>8}{'t':>8}{'부호':>7}"
          f"   {'95% CI':^18}  {'rho(점수)':>10}{'corr(z)':>9}")
    print("-" * 92)
    for name in ARMS:
        if name == "d2p":
            continue
        rho = np.corrcoef(base, S[name])[0, 1]
        cz = np.corrcoef(Z["d2p"], Z[name])[0, 1]
        print(paired(S[name] - base, name) + f"  {rho:>10.2f}{cz:>9.4f}")

    # ── ③ 요인 분해 ──────────────────────────────────────────────────────
    print("\n[③ 요인 분해 — §7.25 가 구조적으로 못 쟀던 것]")
    print(f"{'항':<34}{'쌍체Δ':>9}{'SE':>8}{'t':>8}{'부호':>7}   {'95% CI':^18}")
    print("-" * 92)
    r_main = np.mean([S[f"d{d}r"] - S[f"d{d}p"] for d in DEPTHS], axis=0)
    print(paired(r_main, "잔차 주효과 (깊이 평균)"))
    for lv, tag in (("p", "plain"), ("r", "resid")):
        for d in DEPTHS[1:]:
            print(paired(S[f"d{d}{lv}"] - S[f"d2{lv}"], f"깊이 2→{d}  @{tag}"))
    inter = (S["d6r"] - S["d6p"]) - (S["d2r"] - S["d2p"])
    print(paired(inter, "★ 상호작용 (d6r−d6p)−(d2r−d2p)"))

    # ── ④ β 스윕 (무학습) ────────────────────────────────────────────────
    print("\n[④ β 스윕 — d2p 풀에 팔을 β 로 섞는다 (재학습 0회)]")
    betas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    z0 = mlp.score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA * Z["d2p"], yva)
    print(f"{'팔':<10}" + "".join(f"{b:>9.2f}" for b in betas) + "   beta*")
    for name in ARMS:
        if name == "d2p":
            continue
        row = np.array([mlp.score((1 - MLP_ALPHA) * z_lgb + MLP_ALPHA
                                  * ((1 - b) * Z["d2p"] + b * Z[name]), yva)
                        for b in betas]) - z0
        print(f"{name:<10}" + "".join(f"{v:>9.2f}" for v in row)
              + f"   {betas[int(np.argmax(row))]:.2f}")

    np.savez_compressed(CACHE / f"mlpdepth_f{fold}.npz",
                        **{f"s_{k}": v for k, v in S.items()})
    print(f"\n사전 등록 채점 — P1 잔차주효과 |Δ|<{base.std(ddof=1):.2f} / P2 단조음수 /"
          f" P3 corr>0.99 / P4 상호작용>0 / P5 {'통과' if ok5 else '실패'}")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
