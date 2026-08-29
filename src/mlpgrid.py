"""§7.217 MLP **깊이 × 잔차 × 에폭 × 폭** 격자 — 챔피언 등가 fold 층에서 다시.

base: `hj05_r047f330.zip` (사용자 지정 챔피언, LB 1123.7187284865)
측정층: `champfold.py` 층 = 코어 LGBM 20(챔피언 레시피) + MLP 8(**emb dim 8 + 콜드패치**)
        + CatBoost 12(α0.10) + δ 5축(jointbeta 배포 경로).
층 격차: compact paired · general_lgb 12 · user_fspec 라우팅(예측 가중 ~11%)과
        PROB_SCALE 1.05 가 없다. 크기는 챔피언에서 그대로 안 나온다.
재베이스 계획: 통과 팔은 `export_weights`(W0/W1/W2 하드코딩 + BN 접기)와
        `mlp_infer.MLPPredictor`(BatchNorm/Res 경로 없음)를 일반화한 뒤
        numpy-torch `0.000e+00` 대조를 통과해야 zip 에 들어간다. §7.196④ 그대로다.

■ §7.196 과 무엇이 다른가 — **층이 다르다**
§7.196 은 `mlpreg.py` 층(LGBM 12 · CatBoost 없음 · **pitcher emb dim 4**)에서 쟀다.
§7.209b·§7.211 이 확인했듯 **dim 이 결론을 만든다** (콜드 패치 판정이 dim4↔dim8 에서
뒤집혔다). 챔피언은 dim 8 이므로 여기서는 dim 8 + 콜드 패치로 다시 잰다.
그래서 §7.196 의 캐시는 재사용하지 않는다 — 앵커부터 다른 물건이다.

■ 격자 (사용자 지시: 깊이 plain/resnet · 에폭 낮은 것부터 · 폭 3가지 · β 스윕)
  D 깊이 × 잔차   depth {2,4,6,8} × {plain, resid}   (ep9 · hid128)   d2p = 앵커
  E 에폭 사다리   epochs {5,7,9,12,15} × {d2p, d4r}
                  ⚠️ 에폭마다 **별도 런**이다. `OneCycleLR(total_steps)` 이 epochs 에서
                     나오므로 15에폭 런의 9번째 에폭은 9에폭 런과 다른 모델이다 (§7.196).
  W 폭 3가지      hid {64, 128, 256} @ d4r — §7.196 의 유일 생존 팔 위에서 묻는다
                  (d2p 에서의 wide256 은 §7.26 이 이미 -88 로 쟀다. 같은 걸 또 재지 않는다)
  β 스윕          **8멤버 배포 풀**(emb4×5 + noid×3) 대비. §7.196④ 가 남긴 숙제다 —
                  거기 β 스윕은 `d2p` 5시드 프록시 풀 대비여서 noid 의 다양성 공급이 빠져 있었다.

■ 시드와 검정력 — 미리 적는다 (MEMORY: preregister-power-not-just-thresholds)
시드 3개다 (42·7·2024). §7.198 이 이 층의 `SD(Δ)@5시드 = 5.23/3.13` 을 실측했으므로
3시드 쌍체의 SE 는 대략 그 1.3배다. **±2 규모를 판정할 설계가 아니다.** 이 격자는
"어느 칸이 -10 이상으로 죽는가"를 가르는 스크린이고, 살아남은 칸은 5시드 재측정이 필요하다.

■ 정합성 게이트 (공짜)
`--check` 는 앵커 설정(depth2·plain·hid128·ep9)을 **다시 학습**해 캐시된
`champ_D8C_f{fold}_emb4_42.npz` 와 `max|Δz| = 0` 인지 본다. 0 이 아니면 이 스크립트의
데이터 경로가 `foldchamp.py` 와 다른 것이고, 그러면 아래 전부 무효다.

실행
    .venv/Scripts/python.exe -u src/mlpgrid.py --check 2024
    .venv/Scripts/python.exe -u src/mlpgrid.py --train 2024 2>&1 | tee experiments/mlpgrid_train_f2024.log
    .venv/Scripts/python.exe -u src/mlpgrid.py --train 2023 2>&1 | tee experiments/mlpgrid_train_f2023.log
    .venv/Scripts/python.exe -u src/mlpgrid.py --score  2>&1 | tee experiments/mlpgrid.log
"""
import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import numpy as np

import foldcache
import mlp
import modelabel
import soften
from config import MLP_AUX_LAMBDA, MLP_VARIANTS, ROOT, SOFT_LABEL_F, TRAIN_CSV
from data import load_csv

CACHE = ROOT / "experiments" / "cache"
FOLDS = (2023, 2024)
SEEDS = (42, 7, 2024)                 # 3시드 — 위 "검정력" 절 참조
P_DIM = 8                             # 챔피언 pitcher_id 임베딩 폭 (config 는 아직 4)
ANCHOR = "d2p"                        # = champ_D8C (배포 구성) — 캐시 재사용
TCRIT = 4.303                         # t(0.975, df=2)


def _arm(depth, resid, hid=128, epochs=9):
    return dict(depth=depth, resid=resid, hid=hid, epochs=epochs)


#  비용 오름차순으로 둔다 — 중간에 끊겨도 싼 칸부터 채워져 있게.
ARMS = {}
ARMS["w64r4"] = _arm(4, True, hid=64)                     # W
ARMS["d2p_e5"] = _arm(2, False, epochs=5)                 # E
ARMS["d4r_e5"] = _arm(4, True, epochs=5)
ARMS["d2p_e7"] = _arm(2, False, epochs=7)
ARMS["d2r"] = _arm(2, True)                               # D
ARMS["d4r_e7"] = _arm(4, True, epochs=7)
ARMS["d4p"] = _arm(4, False)
ARMS["d4r"] = _arm(4, True)
ARMS["d2p_e12"] = _arm(2, False, epochs=12)
ARMS["d2p_e15"] = _arm(2, False, epochs=15)
ARMS["w256r4"] = _arm(4, True, hid=256)                   # W
ARMS["d4r_e12"] = _arm(4, True, epochs=12)
ARMS["d6p"] = _arm(6, False)
ARMS["d4r_e15"] = _arm(4, True, epochs=15)
ARMS["d8p"] = _arm(8, False)
ARMS["d6r"] = _arm(6, True)
ARMS["d8r"] = _arm(8, True)

BLOCKS = {
    "D 깊이×잔차": ["d2r", "d4p", "d4r", "d6p", "d6r", "d8p", "d8r"],
    "E 에폭": ["d2p_e5", "d2p_e7", "d2p_e12", "d2p_e15",
               "d4r_e5", "d4r_e7", "d4r_e12", "d4r_e15"],
    "W 폭@d4r": ["w64r4", "w256r4"],
}


def base_cfg():
    """배포 emb4 설정 + pitcher_id 임베딩 폭 8 (챔피언). 나머지는 손대지 않는다."""
    cfg, _ = copy.deepcopy(MLP_VARIANTS["emb4"])
    cfg["emb_dim"] = dict(cfg["emb_dim"], pitcher_id=P_DIM)
    return cfg


def cold_patch(m, dim=P_DIM):
    """미지 슬롯(index 0)을 학습된 나머지의 중심으로 (§7.203·§7.206). foldchamp 와 동일."""
    import torch
    with torch.no_grad():
        hit = [e for e in m.emb if e.weight.shape[1] == dim and e.weight.shape[0] > 100]
        assert len(hit) == 1, f"pitcher 임베딩 특정 실패 ({len(hit)}개)"
        w = hit[0].weight
        b4 = float(w[0].norm())
        w[0] = w[1:].mean(0)
        return b4, float(w[0].norm())


def cache_path(tag, fold, seed):
    if tag == ANCHOR:
        return CACHE / f"champ_D8C_f{fold}_emb4_{seed}.npz"
    return CACHE / f"mg_{tag}_f{fold}_{seed}.npz"


def prepare(df, fold):
    Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    aux, cov = modelabel.aux_labels(tr)
    if SOFT_LABEL_F:
        Xtr, ytr, w, aux, h = soften.soften(tr, Xtr, ytr, w, aux)
    A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
    del Xtr, Xva, art, tr
    gc.collect()
    return A, C, ytr, w, B, Cv, sizes, aux, yva


def train_one(tag, cfg, pack, seed, t0):
    A, C, ytr, w, B, Cv, sizes, aux, yva = pack
    m, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=seed, verbose=False,
                             y_aux=aux, **dict(base_cfg(), **cfg, aux=MLP_AUX_LAMBDA))
    del curve
    b4, af = cold_patch(m)
    z = mlp.infer(m, B, Cv)
    del m
    gc.collect()
    return z, b4, af


def cmd_check(fold):
    """정합성 게이트 — 앵커 설정을 다시 학습해 캐시와 비트 대조."""
    df = load_csv(TRAIN_CSV)
    pack = prepare(df, fold)
    del df
    gc.collect()
    t0 = time.time()
    z, b4, af = train_one("chk", ARMS.get("chk", _arm(2, False)), pack, 42, t0)
    ref = np.load(CACHE / f"champ_D8C_f{fold}_emb4_42.npz")["z"]
    d = float(np.abs(z.astype(np.float64) - ref.astype(np.float64)).max())
    print(f"정합성 게이트 fold {fold} seed 42   max|Δz| = {d:.3e}   "
          f"({'통과' if d == 0.0 else '실패 — 데이터 경로가 foldchamp 와 다르다'})")
    print(f"  ‖emb0[0]‖ {b4:.3f} -> {af:.3f}   ({time.time()-t0:.0f}s)")
    return d == 0.0


def cmd_train(fold):
    todo = [(t, c) for t, c in ARMS.items()
            if any(not cache_path(t, fold, s).exists() for s in SEEDS)]
    print(f"fold {fold}  학습할 팔 {len(todo)}/{len(ARMS)}   시드 {SEEDS}   "
          f"pitcher emb dim {P_DIM} + 콜드패치")
    if not todo:
        print("  전부 캐시 적중 — 건너뜀")
        return
    df = load_csv(TRAIN_CSV)
    pack = prepare(df, fold)
    yva = pack[-1]
    del df
    gc.collect()
    t0 = time.time()
    for tag, cfg in todo:
        for s in SEEDS:
            p = cache_path(tag, fold, s)
            if p.exists():
                continue
            z, b4, af = train_one(tag, cfg, pack, s, t0)
            np.savez_compressed(p, z=z)
            print(f"  [{time.time()-t0:>6.0f}s] {tag:<9} seed {s:<6} "
                  f"단독 {mlp.score(z, yva):>8.1f}   ‖emb0[0]‖ {b4:.3f}->{af:.3f}")
    print(f"fold {fold} 완료  총 {time.time()-t0:.0f}s")


def paired(d, label, width=30):
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0.0
    return (f"{label:<{width}}{d.mean():>9.2f}{se:>8.2f}{t:>8.2f}"
            f"{int((d > 0).sum())}/{len(d)}"
            f"   [{d.mean()-TCRIT*se:+8.2f}, {d.mean()+TCRIT*se:+8.2f}]")


def cmd_score():
    import pandas as pd

    import champfold as CF
    import hillclimb as H
    from config import MLP_ALPHA, TARGET_COL
    from validation import bss, shift_to_prior

    df = load_csv(TRAIN_CSV)
    s = df["season"].to_numpy()
    y_all = df[TARGET_COL].to_numpy(dtype=np.float64)

    print("§7.217 MLP 깊이×잔차×에폭×폭 격자 — 챔피언 등가 fold 층 (dim8 + 콜드패치)")
    print(f"  base = hj05_r047f330.zip · 시드 {SEEDS} · 앵커 {ANCHOR} = 배포 구성(D8C)")
    print("  ⚠️ 라우팅 29파일·PROB_SCALE 미반영 — 크기는 전이 안 된다\n")

    SC, ZM = {}, {}
    for fold in FOLDS:
        st, mm, yv, d, core, cat, noid, ax, tr, w = CF.fold_state(df, s, y_all, fold)
        r = float(yv.mean())
        nz = np.mean(noid, axis=0)

        def blend(zm):
            return ((1 - H.W_CAT) * ((1 - MLP_ALPHA) * core + MLP_ALPHA * zm)
                    + H.W_CAT * cat + ax)

        def sc(z):
            return bss(yv, shift_to_prior(1 / (1 + np.exp(-z)), r))

        for tag in [ANCHOR] + list(ARMS):
            ps = [cache_path(tag, fold, sd) for sd in SEEDS]
            if not all(p.exists() for p in ps):
                continue
            zs = [np.load(p)["z"] for p in ps]
            SC[(fold, tag)] = np.array(
                [sc(blend((0.875 * z + 0.125 * nz)[mm])) for z in zs])
            ZM[(fold, tag)] = np.mean(zs, axis=0)

        have = [t for t in ARMS if (fold, t) in SC]
        base = SC[(fold, ANCHOR)]
        z0pool = (0.875 * ZM[(fold, ANCHOR)] + 0.125 * nz)[mm]
        b0 = sc(blend(z0pool))
        print("=" * 112)
        print(f"fold {fold}   valid {len(yv):,}행   앵커 {ANCHOR} "
              f"시드평균 {base.mean():.2f} (SD {base.std(ddof=1):.2f})  "
              f"풀 {b0:.2f}   팔 {len(have)}/{len(ARMS)}")
        print("=" * 112)
        print(f"{'팔':<30}{'쌍체Δ':>9}{'SE':>8}{'t':>8}{'부호':>5}"
              f"   {'95% CI':^20}{'corr(z)':>9}")
        for blk, tags in BLOCKS.items():
            print(f"-- {blk} " + "-" * (108 - len(blk)))
            for tag in tags:
                if (fold, tag) not in SC:
                    continue
                cz = np.corrcoef(ZM[(fold, ANCHOR)], ZM[(fold, tag)])[0, 1]
                print(paired(SC[(fold, tag)] - base, tag) + f"{cz:>9.4f}")

        # β 스윕 — **8멤버 배포 풀** 대비 (§7.196④ 의 숙제)
        betas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        print(f"\n[β 스윕 — 배포 풀(emb4×3 + noid×3) 의 emb4 몫을 β 로 교체]  기준 {b0:.2f}")
        print(f"{'팔':<12}" + "".join(f"{b:>9.2f}" for b in betas) + f"{'β*':>8}")
        for tag in have:
            row = np.array([
                sc(blend((0.875 * ((1 - b) * ZM[(fold, ANCHOR)] + b * ZM[(fold, tag)])
                          + 0.125 * nz)[mm])) - b0 for b in betas])
            print(f"{tag:<12}" + "".join(f"{v:>+9.2f}" for v in row)
                  + f"{betas[int(np.argmax(row))]:>8.2f}")
        print()

    # ── 양 fold 요약 ────────────────────────────────────────────────────
    print("=" * 112)
    print("양 fold 요약 — 부호 일치 팔만 후보다 (CLAUDE.md §7)")
    print("=" * 112)
    print(f"{'팔':<12}{'f2023 Δ':>10}{'f2024 Δ':>10}{'부호':>6}{'크기비':>9}")
    for tag in ARMS:
        if (2023, tag) not in SC or (2024, tag) not in SC:
            continue
        a = (SC[(2023, tag)] - SC[(2023, ANCHOR)]).mean()
        b = (SC[(2024, tag)] - SC[(2024, ANCHOR)]).mean()
        ok = "같음" if a * b > 0 else "엇갈림"
        ratio = a / b if abs(b) > 1e-9 else float("nan")
        print(f"{tag:<12}{a:>+10.2f}{b:>+10.2f}{ok:>6}{ratio:>9.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int)
    ap.add_argument("--train", type=int)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LGA_THREADS", "4")))
    a = ap.parse_args()
    mlp.THREADS = a.threads
    if a.check:
        cmd_check(a.check)
    elif a.train:
        cmd_train(a.train)
    elif a.score:
        cmd_score()
    else:
        ap.error("--check / --train / --score 중 하나")
