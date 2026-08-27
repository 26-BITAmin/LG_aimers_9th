"""정규화 노브 재측정 — **배포 등가 하네스 위에서** (§7.172 재실행 + 확장).

§7.172 는 `ablate19.py`(§7.18 시절 코드)를 물려받아 **`soften`(soft label, s23·LB +3.09)
과 `aux_labels`(보조 헤드 λ=1.0), `contam_axes`/`center_asof` 를 전부 빠뜨린 채** 쟀다.
약 50 LB점어치 스택이 빠진 프록시였다. 혜진 13-4 가 *"프록시의 규제 수준이 배포와 다르면
부호까지 뒤집힐 수 있다"* 를 실측했으므로(평활 +13.67 → −11.54) 그 판정은 무효다.

soft label 은 **라벨 노이즈를 줄이는 장치**이고 `min_data_in_leaf`·`lambda_l2` 도
**라벨 노이즈 과적합을 막는 장치**다. 대체재끼리라 하나를 빼고 다른 하나를 재면
상호작용이 통째로 빠진다. 원격한 가능성이 아니라 정면으로 예상되는 교란이다.

**이번에는 `foldcache.fold_data` 를 직접 import 해서 쓴다** — 유지보수되는 배포 등가 경로다.

#### 양성 대조군

§7.172 의 널들은 계측기가 살아 있다는 증거가 없으면 무의미하다. 그래서 **크기를 아는
조작**을 팔로 넣는다: `soft_label_f` OFF. s23 에서 LB +3.09 로 채택됐고, 과거 이걸 끄니
fold2023 이 −1255 로 무너진 기록이 있다. 이 팔이 크게 음수로 나와야 나머지 널이 의미를 갖는다.
못 나오면 **널 전체가 무효**이고 하네스부터 고쳐야 한다.

실행:  .venv/Scripts/python.exe -u src/regloose2.py 2024
"""
import gc
import re
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
                    N_ROUNDS, TRAIN_CSV)
from data import load_csv

mlp.THREADS = 8
CACHE = Path(__file__).parent.parent / "experiments" / "cache"
SEEDS = (42, 7, 2024, 1234, 31337, 777, 31415)          # n=7 쌍체
EMB4 = {"pitcher_id": 4, "batter_id": 4, "pitcher_team_id": 4, "batter_team_id": 4,
        "base_state": 4, "hand_matchup": 4, "count_state": 4}
CFG_EMB4 = dict(epochs=9, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=True, emb_dim=EMB4)
CFG_NOID = dict(epochs=6, hid=128, lr=2e-3, dropout=0.3, wd=1e-2, use_id_emb=False, emb_dim=EMB4)

# soft=True 팔들은 배포와 같은 학습셋. soft=False 는 양성 대조군.
ARMS = {
    "기준":              dict(soft=True,  over={}),
    "md 500":            dict(soft=True,  over=dict(min_data_in_leaf=500)),
    "l2 10":             dict(soft=True,  over=dict(lambda_l2=10.0)),
    "l2 100 [대조]":      dict(soft=True,  over=dict(lambda_l2=100.0)),
    "soft OFF [양성대조]": dict(soft=False, over={}),
}


def build_mlp(A, C, ytr, w, B, Cv, sizes, aux, yva, tag, t0):
    """5멤버 MLP. 팔 사이에서 고정되므로 쌍체 차분에서 상쇄된다."""
    f = CACHE / f"regloose2_mlp_{tag}.npz"
    if f.exists():
        print(f"  [{time.time()-t0:.0f}s] MLP({tag}) 재사용")
        return np.load(f)["z"]
    zs = []
    for cfg, sd in ((CFG_EMB4, (42, 7, 2024)), (CFG_NOID, (42, 7))):
        for s in sd:
            _, curve = mlp.train_mlp(A, C, ytr, w, B, Cv, sizes, seed=s, verbose=False,
                                     y_aux=aux, **dict(cfg, aux=MLP_AUX_LAMBDA))
            zs.append(curve[-1])
    z = np.mean(zs, axis=0)
    np.savez_compressed(f, z=z)
    print(f"  [{time.time()-t0:.0f}s] MLP({tag}) 5멤버 단독 {mlp.score(z, yva):.1f}")
    return z


def main(fold):
    t0 = time.time()
    df = load_csv(TRAIN_CSV)
    Xtr0, ytr0, w0, Xva, yva, art, tr = foldcache.fold_data(df, fold)
    del df; gc.collect()
    aux0, cov = modelabel.aux_labels(tr)
    print(f"[{time.time()-t0:.0f}s] fold={fold}  train {Xtr0.shape}  valid {Xva.shape}"
          f"  보조라벨 복원 {cov:.1%}")

    # soft ON / OFF 두 학습셋
    Xs, ys, ws, auxs, h = soften.soften(tr, Xtr0, ytr0, w0, aux0)
    print(f"[{time.time()-t0:.0f}s] soft label: {len(Xtr0):,} → {len(Xs):,}행  h={h}")
    SETS = {True: (Xs, ys, ws, auxs), False: (Xtr0, ytr0, w0, aux0)}

    Z_MLP = {}
    for soft in (True, False):
        X, y, w, aux = SETS[soft]
        A, C, B, Cv, _, sizes = mlp.encode(X, Xva, art)
        Z_MLP[soft] = build_mlp(A, C, y, w, B, Cv, sizes, aux, yva,
                                f"f{fold}_soft{int(soft)}", t0)
        del A, C, B, Cv; gc.collect()

    # ⚠️ 캐시 키가 **정말로 갈렸는지** 먼저 검정한다. 첫 실행에서 "l2 10" 과
    #    "l2 100 [대조]" 가 둘 다 "l2" 로 접혀 대조군이 후보의 캐시를 읽었다 —
    #    Δ·SE·t·양수 개수가 소수점 끝까지 같아서야 알아챘다. 값으로 눈치채기 전에
    #    키로 막는다.
    slugs = {n: re.sub(r"[^\w.]+", "_", n).strip("_") for n in ARMS}
    assert len(set(slugs.values())) == len(ARMS), f"캐시 키 충돌: {slugs}"
    print(f"[키 검정] {len(ARMS)}개 팔 -> 슬러그 {len(set(slugs.values()))}개, 충돌 없음")

    res = {}
    for name, spec in ARMS.items():
        # ⚠️ 키는 팔 이름 **전체**에서 만든다. name.split()[0] 을 쓰면
        #    "l2 10" 과 "l2 100 [대조]" 가 둘 다 "l2" 로 접혀 같은 파일을 가리킨다
        #    (실제로 그렇게 났다 — 대조군이 후보의 캐시를 읽어 값이 완전히 같았다).
        slug = slugs[name]
        f = CACHE / f"regloose2_arm_f{fold}_{slug}_{spec['soft']}.npz"
        if f.exists():
            res[name] = np.load(f)["s"]
            print(f"[{time.time()-t0:.0f}s] {name:<16} 재사용 {res[name].mean():8.2f}")
            continue
        X, y, w, _ = SETS[spec["soft"]]
        z_mlp, per_seed = Z_MLP[spec["soft"]], []
        for s in SEEDS:
            z_s = np.zeros(len(Xva))
            for vname, vover in ENSEMBLE_VARIANTS.items():
                eff = {k: v for k, v in spec["over"].items() if k not in vover}
                p = dict(FINAL_LGB_PARAMS); p.update(vover); p.update(eff)
                p.update(seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s)
                bst = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=N_ROUNDS)
                z_s += bst.predict(Xva, raw_score=True)
                del bst
            z_s /= len(ENSEMBLE_VARIANTS)
            per_seed.append(mlp.score((1 - MLP_ALPHA) * z_s + MLP_ALPHA * z_mlp, yva))
            gc.collect()
        res[name] = np.array(per_seed)
        np.savez_compressed(f, s=res[name])       # 팔 단위 저장 — OOM 로 끊겨도 여기까진 산다
        print(f"[{time.time()-t0:.0f}s] {name:<16} 시드별 {res[name].mean():8.2f}"
              f" ± {res[name].std(ddof=1):.2f}")

    base = res["기준"]
    print(f"\n{'팔':<16}{'쌍체평균Δ':>11}{'SE':>8}{'t':>8}{'양수':>7}  95% CI")
    print("-" * 68)
    for name, s in res.items():
        if name == "기준":
            continue
        d = s - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0.0
        print(f"{name:<16}{d.mean():>11.2f}{se:>8.2f}{t:>8.2f}{f'{(d>0).sum()}/7':>7}"
              f"  [{d.mean()-2.447*se:+.2f}, {d.mean()+2.447*se:+.2f}]")
    print("\n※ 양성대조(soft OFF)가 크게 음수여야 나머지 널이 의미를 갖는다.")
    np.savez_compressed(CACHE / f"regloose2_f{fold}.npz", **res)
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
