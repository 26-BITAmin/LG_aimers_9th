"""§7.208 — 챔피언 **레시피**를 fold 로 재학습해 정직한 홀드아웃 층을 만든다.

§7.207 이 확정한 것: 챔피언 zip 은 2019~2024 전부로 학습돼 **홀드아웃이 없다.**
그래서 "챔피언 위에서 재라"는 산출물이 아니라 **레시피**로 해야 한다.

챔피언의 우리 대비 실질 차이 (§7.206 실물 대조):
    ① `pitcher_id` 임베딩 4 → **8**            ← 재현 가능 (config 한 줄)
    ② 콜드스타트: emb0[0] ← mean(emb0[1:])    ← 재현 가능 (텐서 한 행)
    ③ compact·general_lgb·fspec·routing        ← 채연 계보, **학습 코드 없음**
    나머지 76/82 파일은 우리 것과 바이트 동일하다.

이 스크립트는 ①②를 fold 로 재학습한다 (train < fold). fold2024 는 직전 이력 대비
**신규 투수 행이 19.86%** 라 (§7.203) 콜드스타트 상황이 마스킹 없이 자연 재현된다.

팔 3개, 같은 시드·같은 데이터:
    D4      emb_dim(pitcher_id)=4  — 현 배포 등가 (대조군)
    D8      emb_dim(pitcher_id)=8  — 챔피언 ①
    D8C     D8 + 콜드 패치         — 챔피언 ①②

⚠️ `foldcache._key()` 는 MLP_VARIANTS 를 지문에 넣으므로 캐시가 자동으로 갈린다.
   산출물 접두는 `champ_` 로 따로 둔다 (병렬 세션과 공존).

실행: .venv/Scripts/python.exe -u src/foldchamp.py 2>&1 | tee experiments/foldchamp.log
"""
import copy
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

import foldcache
import mlp
import modelabel
import soften
from config import (MLP_AUX_LAMBDA, MLP_SEEDS, MLP_VARIANTS, ROOT, SOFT_LABEL_F,
                    TRAIN_CSV)
from data import load_csv
from validation import bss

CACHE = ROOT / "experiments" / "cache"
FOLDS = (2023, 2024)
ARMS = ("D4", "D8", "D8C")


def variants(dim):
    v = copy.deepcopy(dict(MLP_VARIANTS))
    cfg, n = v["emb4"]
    cfg = dict(cfg)
    cfg["emb_dim"] = dict(cfg["emb_dim"])
    cfg["emb_dim"]["pitcher_id"] = dim
    v["emb4"] = (cfg, n)
    return v


def cold_patch(m, dim):
    """미지 슬롯(index 0)을 학습된 나머지의 중심으로 접는다 (§7.203·§7.206)."""
    import torch
    with torch.no_grad():
        hit = [e for e in m.emb if e.weight.shape[1] == dim and e.weight.shape[0] > 100]
        assert len(hit) == 1, f"pitcher 임베딩 특정 실패 ({len(hit)}개)"
        w = hit[0].weight
        before = float(w[0].norm())
        w[0] = w[1:].mean(0)
        return before, float(w[0].norm())


def main():
    df = load_csv(TRAIN_CSV)
    t0 = time.time()
    for fold in FOLDS:
        need = [(a, s) for a in ARMS for s in MLP_SEEDS
                if not (CACHE / f"champ_{a}_f{fold}_emb4_{s}.npz").exists()]
        if not need and (CACHE / f"champ_noid_f{fold}.npz").exists():
            print(f"fold {fold} 캐시 적중 — 건너뜀")
            continue
        print(f"\n=== fold {fold} ===")
        Xtr, ytr, w, Xva, yva, art, tr = foldcache.fold_data(df, fold)
        aux, cov = modelabel.aux_labels(tr)
        if SOFT_LABEL_F:
            Xtr, ytr, w, aux, h = soften.soften(tr, Xtr, ytr, w, aux)
            print(f"  soft label h={h}  →{len(Xtr):,}행 · 보조라벨 {cov:.1%}")
        np.savez_compressed(CACHE / f"champ_y_f{fold}.npz", y=yva)

        # noid 팔은 dim 과 무관 — 한 번만 학습해 세 팔이 공유한다
        for dim, arms in ((4, ["D4"]), (8, ["D8", "D8C"])):
            V = variants(dim)
            A, C, B, Cv, _, sizes = mlp.encode(Xtr, Xva, art)
            for vname, (cfg, nseed) in V.items():
                if vname == "noid" and (CACHE / f"champ_noid_f{fold}.npz").exists():
                    continue
                zs = []
                for s in MLP_SEEDS[:nseed]:
                    tag = f"champ_{arms[0]}_f{fold}_{vname}_{s}.npz"
                    if vname == "noid":
                        pass
                    elif (CACHE / tag).exists() and (
                            len(arms) == 1 or (CACHE / tag.replace(arms[0], arms[1])).exists()):
                        continue
                    m, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s,
                                             verbose=False, y_aux=aux,
                                             **dict(cfg, aux=MLP_AUX_LAMBDA))
                    z = curve[-1]
                    if vname == "noid":
                        zs.append(z)
                    else:
                        np.savez_compressed(CACHE / tag, z=z)
                        print(f"  {arms[0]} {vname}_{s} 단독 {mlp.score(z, yva):.1f}"
                              f"  ({time.time()-t0:.0f}s)")
                        if len(arms) > 1:            # 콜드 패치 팔
                            b4, af = cold_patch(m, dim)
                            zc = mlp.infer(m, B, Cv)
                            np.savez_compressed(
                                CACHE / tag.replace(arms[0], arms[1]), z=zc)
                            print(f"  {arms[1]} {vname}_{s} 단독 {mlp.score(zc, yva):.1f}"
                                  f"   ‖emb0[0]‖ {b4:.3f}→{af:.3f}")
                    del m
                    gc.collect()
                if vname == "noid" and zs:
                    np.savez_compressed(CACHE / f"champ_noid_f{fold}.npz",
                                        z=np.stack(zs))
                    print(f"  noid {len(zs)}개 저장  ({time.time()-t0:.0f}s)")
            del A, C, B, Cv
            gc.collect()
        del Xtr, Xva
        gc.collect()

    print("\n=== 팔별 MLP 8멤버 평균 단독 BSS ===")
    for fold in FOLDS:
        y = np.load(CACHE / f"champ_y_f{fold}.npz")["y"]
        noid = np.load(CACHE / f"champ_noid_f{fold}.npz")["z"]
        print(f"  fold {fold}")
        for a in ARMS:
            e = [np.load(CACHE / f"champ_{a}_f{fold}_emb4_{s}.npz")["z"]
                 for s in MLP_SEEDS]
            zm = np.mean(list(e) + list(noid), axis=0)
            print(f"    {a:<5} {mlp.score(zm, y):>8.1f}"
                  f"   (emb4 5멤버 평균 {mlp.score(np.mean(e,axis=0), y):.1f})")


if __name__ == "__main__":
    main()
