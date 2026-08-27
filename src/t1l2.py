"""§7.176 후속 — s41 코어에 **`lambda_l2` 30 → 10** 을 얹는다.

§7.176 이 배포 등가 하네스(fold2024, n=7 쌍체)에서 얻은 유일한 생존 축이다.

    λ2 10  (완화)   Δ +0.26  t +2.83  6/7  CI [+0.03, +0.48]
    λ2 100 (조임)   Δ -0.75  t -5.63  0/7  ← 대조군이 반대 부호
    정점 적합       λ2* = 8.56 → 우리가 잰 10 이 곧 정점

남은 불확실성은 **앙상블 규모**다. 측정은 복제당 4변형×1시드(M4), 배포는 4×5(M20).
`wd` 가 정확히 1/M 로 죽은 전례(§7.176③)가 있어 λ2 도 그럴 수 있다. 그러나 그것은
가정이지 측정이 아니고, **LB 는 노이즈 0 · 하방 0** 이다 (§7.161b-0). 90분짜리 M 곡선
대신 슬롯 한 장으로 답한다.

s41 zip 을 베이스로 **코어 lgbm 20 개만** 교체한다 — 나머지 54 개 파일은 손대지 않는다.
`extra` 변형은 이미 `lambda_l2=10.0` 이라 오버라이드가 건너뛴다 (3/4 변형에만 걸린다).

실행:  .venv/Scripts/python.exe -u src/t1l2.py
"""
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
T1 = ROOT / "t1work"
ASCII = Path("D:/t1eval")
sys.path.insert(0, str(Path(__file__).parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import lightgbm as lgb
import numpy as np

from config import (ENSEMBLE_VARIANTS, FINAL_LGB_PARAMS, N_ROUNDS,
                    RECENCY_HALF_LIFE, TRAIN_CSV)
from data import load_csv, recency_weight
from t1core import SEEDS, VARIANTS, build, t1_features

L2 = 10.0
REF_SEASON = 2024
ARM = "G_full_aug_l2"
S41 = ROOT / "s41_t1core_aug.zip"
OUT = ROOT / "s42_l2x10.zip"


def _l2_of(path):
    """모델 파일의 [parameters] 블록에서 lambda_l2 를 읽는다 — 복사본이 흉내낼 수 없는 지문."""
    blk = path.read_text(encoding="utf-8").split("parameters:")[1]
    blk = blk.split("end of parameters")[0]
    for ln in blk.splitlines():
        ln = ln.strip()
        if ln.startswith("[lambda_l2:"):
            return float(ln[1:-1].split(":")[1])
    return None


def main():
    t0 = time.time()
    done = []
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    X = t1_features(df, art, cols)
    w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=REF_SEASON)
    print(f"[{time.time()-t0:.0f}s] 전 시즌 {len(df):,}행 · 피처 {X.shape}")

    dst = ASCII / ARM
    if not (dst / "model" / "artifacts.joblib").exists():
        shutil.copytree(T1, dst, dirs_exist_ok=True)   # 없을 때만 새로 깐다 (재개 보존)
    rng = np.random.default_rng(777)                 # t1full 과 같은 시드 = 같은 합성 행
    X2, y2, w2, init, k = build(df, X, w, True, rng)
    print(f"[{time.time()-t0:.0f}s] 증강 {len(X):,} → {len(X2):,} (합성 {k:,})")
    mf = dst / "_l2_trained.json"
    manifest = set(json.loads(mf.read_text())) if mf.exists() else set()
    if manifest:
        print(f"  매니페스트 재개: 완료 {len(manifest)}개")
    del X, df

    hit = []
    for v in VARIANTS:
        over = ENSEMBLE_VARIANTS[v]
        eff = {} if "lambda_l2" in over else {"lambda_l2": L2}   # 변형 값이 이긴다
        if eff:
            hit.append(v)
        for sd in SEEDS:
            # ⚠️ 재개 키는 파일 **존재**가 아니라 파일이 스스로 밝히는 λ2 다.
            #    T1 원본을 통째로 복사해두므로 20개가 전부 이미 존재한다 —
            #    존재로 판단하면 미학습 파일을 학습된 것으로 오인한다.
            #    (12:07 OOM 으로 wide 31337 에서 끊겼을 때 이게 필요했다.)
            key = f"{v}_{sd}"
            f = dst / "model" / f"lgbm_{v}_{sd}.txt"
            # ⚠️ 재개 키는 **내가 썼다는 기록**이어야 한다. λ2 값으로 판단하면
            #    extra 변형(원래부터 lambda_l2=10)에서 T1 원본을 "학습 완료"로
            #    오인해 **증강 안 된 모델을 제출물에 싣는다.** 실제로 그럴 뻔했다.
            #    바꾸는 축의 값은 재개 키가 될 수 없다 — 원본과 겹칠 수 있으므로.
            if key in manifest:
                done.append(key)
                continue
            p = dict(FINAL_LGB_PARAMS, **over, **eff, seed=sd, bagging_seed=sd,
                     feature_fraction_seed=sd, data_random_seed=sd)
            b = lgb.train(p, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                          num_boost_round=N_ROUNDS)
            (dst / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                b.model_to_string(), encoding="utf-8", newline="\n")   # LF (§7.173③)
            del b
        print(f"[{time.time()-t0:.0f}s]   {v:<6} {len(SEEDS)}시드  "
              f"λ2={'10 (적용)' if eff else str(over['lambda_l2']) + ' (변형 고정)'}")
    print(f"  → λ2 오버라이드가 걸린 변형: {hit} ({len(hit)}/4)")
    if done:
        print(f"  → 재개: 이미 λ2 일치라 건너뛴 멤버 {len(done)}개 {done}")

    # ── 검증 ① 모델 파일이 스스로 밝히는 λ2 ────────────────────────
    print("\n검증① 배포될 모델이 실제로 어떤 λ2 로 학습됐나")
    for v in VARIANTS:
        txt = (dst / "model" / f"lgbm_{v}_42.txt").read_text(encoding="utf-8")
        blk = txt.split("parameters:")[1].split("end of parameters")[0]
        got = [l.strip()[1:-1].split(":")[1].strip() for l in blk.strip().split("\n")
               if l.strip().startswith("[lambda_l2:")][0]
        print(f"    lgbm_{v}_42.txt  lambda_l2 = {got}")

    # ── 패키징: s41 에서 코어 20 개만 교체 ──────────────────────────
    with zipfile.ZipFile(S41) as z:
        names = z.namelist()
    swap = {f"model/lgbm_{v}_{sd}.txt" for v in VARIANTS for sd in SEEDS}
    assert len(swap) == 20, len(swap)
    if OUT.exists():
        OUT.unlink()
    n_sw = 0
    with zipfile.ZipFile(S41) as src, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for n in names:
            if n in swap:
                out.writestr(n, (dst / n).read_bytes()); n_sw += 1
            else:
                out.writestr(n, src.read(n))
    assert n_sw == 20, n_sw
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"(엔트리 {len(names)}개 중 {n_sw}개 교체)")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
