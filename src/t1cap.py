"""§7.182 — s45 빌드: 챔피언(s43_cat12) 의 코어 LGBM 20 을 **num_leaves ×2** 로 교체

§7.181 (t1knob) 실측: fold2024 쌍체 CAP2 **+1.72 [+0.69, +2.78] P=1.000**,
독립 시드 B=4000 재확인 [+0.68, +2.78] P=0.9992. 용량-반응 대칭(CAPH −1.90),
F/R 분산(+1.8/+1.7), std(p) 불변 — §7.18 "재튜닝 9전 9패"의 T1 계보 역전.

레시피 (§7.175 t1full F_full_aug 와 동일, num_leaves 만 ×2)
    학습: 전 시즌 2019~2024 · SMOTE 증강 ON (rng 777) · soften · init_score
    변형별 num_leaves: base/extra/lowff 31→62 · wide 127→254 (min_data 유지)
    나머지 파일(카탈로그 74개 중 cat12·MLP·compact·fspec·predict.py 등) 전부
    챔피언 원본 유지 — LB 변화 = 코어 용량분.

레벨 재보정 (§7.178 ② — T1 계보는 logit_delta 가 동결이라 코어가 바뀌면 깨진다)
    2024 전 행 프록시로 챔피언 평균 vs 신규 평균을 재고, PROB_SHIFT 를
    (챔피언 shift − Δmean) 으로 패치해 평균을 소수 9자리까지 일치시킨다.
    PROB_SHIFT 는 확률공간 가산이라 뺄셈이 정확히 평균을 정렬한다 (클리핑 없음 가정,
    사후 검증). 챔피언 shift 값은 하드코딩하지 않고 zip 의 predict.py 에서 파싱한다
    ([[audit-inherited-constants]]).

검증
    ⓐ SMOTE k > 0 · dup 행 수 출력 (§7.175 와 대조)
    ⓑ 신규 zip 파일 목록 = 챔피언 목록, 교체는 lgbm 20 + predict.py 만
    ⓒ 프록시 2024 전 행: max|Δp| > 0 (코어 실제 교체) · 패치 후 평균차 < 1e-9

실행:  .venv/Scripts/python.exe -u src/t1cap.py > experiments/t1cap.log 2>&1
"""
import re
import shutil
import subprocess
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
from t1core import SEEDS, VARIANTS, WORKER, build, t1_features

CRLF = chr(13) + chr(10)
LF = chr(10)
REF_SEASON = 2024
BASE_ZIP = ROOT / "s43_cat12.zip"            # 현 챔피언 1118.4706009389
OUT = ROOT / "s45_cap2.zip"
REF_DIR = ASCII / "S43REF"                   # 챔피언 압축 해제본 (프록시 기준)
NEW_DIR = ASCII / "CAP2F"                    # 신규 팔 (챔피언 + 코어 교체)
LEAF_MULT = 2.0


def leaf2(v):
    nl = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v]).get("num_leaves", 31)
    return max(2, int(round(nl * LEAF_MULT)))


def extract(zp, dst):
    if (dst / "predict.py").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dst)


def proxy(dname, out_npz):
    """2024 전 행을 해당 디렉터리의 Predictor 로 예측 (t1core WORKER 재사용)."""
    f = ROOT / "experiments" / "cache" / out_npz
    if not f.exists():
        wk = ROOT / "experiments" / "cache" / "_t1cap_worker.py"
        wk.write_text(WORKER.replace("{csv}", str(TRAIN_CSV).replace("\\", "/")),
                      encoding="utf-8", newline="\n")
        r = subprocess.run([sys.executable, str(wk), dname, str(f)],
                           capture_output=True, text=True)
        if not f.exists():
            print(f"  ❌ {dname}: {r.stderr.strip()[-400:]}")
            sys.exit(1)
    return np.load(f)["p"]


def main():
    t0 = time.time()
    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()

    # ── 1. 코어 20 재학습 (전 시즌 · 증강 ON · leaves ×2) ────────────
    extract(BASE_ZIP, REF_DIR)
    need = [NEW_DIR / "model" / f"lgbm_{v}_{sd}.txt" for v in VARIANTS for sd in SEEDS]
    if all(f.exists() for f in need) and (NEW_DIR / "_cap2_done").exists():
        print("  코어 재사용 (_cap2_done)")
    else:
        df = load_csv(TRAIN_CSV)                     # 전 시즌 2019~2024
        X = t1_features(df, art, cols)
        w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE,
                           ref_season=REF_SEASON)
        X2, y2, w2, init, k = build(df, X, w, True, np.random.default_rng(777))
        assert k > 0, "증강이 안 걸렸다"
        print(f"[{time.time()-t0:.0f}s] 학습 {len(df):,} → {len(X2):,} (SMOTE {k:,})")
        del X, df
        shutil.copytree(REF_DIR, NEW_DIR, dirs_exist_ok=True)
        for v in VARIANTS:
            # ⚠️ wide 는 자체 num_leaves 를 가진다 — 키워드 인자로 주면 중복 TypeError
            p = dict(FINAL_LGB_PARAMS, **ENSEMBLE_VARIANTS[v])
            p["num_leaves"] = leaf2(v)
            for sd in SEEDS:
                p2 = dict(p, seed=sd, bagging_seed=sd,
                          feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p2, lgb.Dataset(X2, y2, weight=w2, init_score=init),
                              num_boost_round=N_ROUNDS)
                # LF — CRLF 는 model_file= 을 죽인다 (§7.173 ③)
                (NEW_DIR / "model" / f"lgbm_{v}_{sd}.txt").write_text(
                    b.model_to_string(), encoding="utf-8", newline="\n")
                del b
            print(f"[{time.time()-t0:.0f}s]   {v} nl={p['num_leaves']} {len(SEEDS)}시드")
        del X2, y2, w2, init
        (NEW_DIR / "_cap2_done").write_text("ok", encoding="utf-8")

    # ── 2. 프록시 평균 정렬 → PROB_SHIFT 재보정 ─────────────────────
    p_ref = proxy("S43REF", "t1cap_ref.npz")
    p_new = proxy("CAP2F", "t1cap_new.npz")
    dmax = float(np.abs(p_new - p_ref).max())
    dmean = float(p_new.mean() - p_ref.mean())
    assert dmax > 0, "코어 교체가 반영되지 않았다"
    print(f"[{time.time()-t0:.0f}s] 프록시 2024 {len(p_ref):,}행  "
          f"max|Δp|={dmax:.5f}  평균|Δp|={np.abs(p_new-p_ref).mean():.5f}  "
          f"Δmean={dmean:+.9f}  (레벨 비용 {1e5*dmean**2/0.2497:.3f}점)")

    pred_path = NEW_DIR / "predict.py"
    src = pred_path.read_text(encoding="utf-8").replace(CRLF, LF)
    m = re.search(r"PROB_SHIFT\s*=\s*([0-9.eE+-]+)", src)
    assert m, "PROB_SHIFT 를 찾지 못했다"
    shift_old = float(m.group(1))
    shift_new = shift_old - dmean
    src2 = src[:m.start(1)] + f"{shift_new:.9f}" + src[m.end(1):]
    assert src2 != src
    pred_path.write_text(src2, encoding="utf-8", newline="\n")
    print(f"  PROB_SHIFT {shift_old:.9f} → {shift_new:.9f}  (챔피언 파싱값 기준)")

    # 패치 검증 — 캐시 무효화 후 재예측
    (ROOT / "experiments" / "cache" / "t1cap_new2.npz").unlink(missing_ok=True)
    p_new2 = proxy("CAP2F", "t1cap_new2.npz")
    d2 = float(p_new2.mean() - p_ref.mean())
    print(f"  패치 후 Δmean = {d2:+.12f}  {'✅' if abs(d2) < 1e-9 else '⚠️ 1e-9 초과'}")
    assert abs(d2) < 1e-7, "평균 정렬 실패"

    # ── 3. 패키징 — 챔피언 목록 그대로, lgbm 20 + predict.py 만 교체 ──
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
    core_names = {f"model/lgbm_{v}_{sd}.txt" for v in VARIANTS for sd in SEEDS}
    assert core_names <= set(names), "챔피언 zip 에 코어 파일명이 없다"
    if OUT.exists():
        OUT.unlink()
    replaced = 0
    with zipfile.ZipFile(BASE_ZIP) as zin, \
         zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for n in names:
            if n in core_names:
                zout.write(NEW_DIR / n, n); replaced += 1
            elif n == "predict.py":
                zout.writestr(n, src2); replaced += 1
            else:
                zout.writestr(n, zin.read(n))
    assert replaced == len(core_names) + 1, replaced
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"(파일 {len(names)}개 · 교체 {replaced}개 = 코어 20 + predict.py)")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
