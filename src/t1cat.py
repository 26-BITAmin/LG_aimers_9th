"""s41r081f375(현 챔피언) 스택에 **CatBoost 12멤버**를 이식한다.

s41 계보의 스택에는 CatBoost 가 **통째로 없다.** 우리 스택에서는 s37 에서 붙어
**LB +1.0854** 를 냈고(1109.26 → 1110.35), 그때의 기저(1109 대)가 지금(1117.9)과
크게 다르지 않다. §7.175 가 같은 형태의 이식(증강)을 우리 +0.79 → T1 +0.88,
**배율 1.1** 로 성공시켰다.

시드를 더 얹는 것과 **모델 계열을 더 얹는 것**은 다르다 — 전자는 분산만 줄여
앙상블과 겹치지만(§7.176a 의 `wd` 가 정확히 1/M 로 죽었다), CatBoost 는 순서형
부스팅·대칭 트리·다른 범주 처리라 **편향 축이 다르다.**

#### 레시피는 s37/s40 과 동일하게 고정한다

  · 입력: **LGBM 과 완전히 같은 X/y/w + init_score(baseline)** (train.py:257)
  · ⛔ **증강은 걸지 않는다.** 메모리 [augmentation-targets-learners-not-just-rows] —
    전부 걸어 −4.4, LGBM 에만 걸어 +1.4. s40 행도 "LGBM 멤버에만 SMOTE 증강" 이다.
    s41 코어(증강 O)는 그대로 두고 CatBoost 만 무증강으로 얹는다.
  · CPU 고정 (train.py:259 — GPU 는 f2023 핸디캡 −9.1 vs CPU +10.3)
  · 블렌드: fspec **이후**, δ 보정 **이전**, 전 행에 α=0.10

베이스는 s41 이 아니라 **현 챔피언 `s41r081f375.zip`** 이다 — 새 후보가
"챔피언 + CatBoost" 가 되어야 LB 변화가 곧 CatBoost 분이 된다.

실행:  .venv/Scripts/python.exe -u src/t1cat.py
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

from config import (CAT_ALPHA, CAT_ITERS, CAT_PARAMS, CAT_SEEDS, CAT_VARIANTS,
                    RECENCY_HALF_LIFE, TRAIN_CSV)
from data import load_csv, recency_weight
from t1core import build, t1_features

CRLF = chr(13) + chr(10)
LF = chr(10)
REF_SEASON = 2024
ARM = "H_cat12"
BASE_ZIP = ROOT / "s41r081f375.zip"          # 현 챔피언 1117.9171927312
OUT = ROOT / "s43_cat12.zip"

BLEND_SRC = """        # 사후 로짓 보정 (투수 × 상황 δ)"""
BLEND_NEW = """        # ── CatBoost 12멤버 블렌드 (우리 스택 §7.142 이식) ──────────
        # fspec **이후**, δ 보정 **이전**. 전 행에 섞는다 (라우팅이 아니다).
        # 각 멤버는 자기 행만 본다 — test 행 간 정보는 쓰지 않는다.
        if self.cat_alpha > 0:
            from catboost import CatBoostClassifier, Pool
            if not self.cats:
                for tag in self.cat_members:
                    m = CatBoostClassifier()
                    m.load_model(os.path.join(self.model_dir, f"cat_{tag}.cbm"))
                    self.cats.append(m)
            pool = Pool(X)
            zc = np.zeros(len(X))
            for m in self.cats:
                zc += m.predict(pool, prediction_type="RawFormulaVal")
            zc /= len(self.cats)
            z = (1.0 - self.cat_alpha) * z + self.cat_alpha * zc

        # 사후 로짓 보정 (투수 × 상황 δ)"""

INIT_SRC = """        self.fspec_alpha = float(bundle.get("fspec_alpha") or 0.0)"""
INIT_NEW = """        # CatBoost 이식 — 멤버 목록은 model/cat_members.json 이 진실이다.
        self.model_dir = model_dir
        cm = os.path.join(model_dir, "cat_members.json")
        if os.path.exists(cm):
            with open(cm, encoding="utf-8") as f:
                d = json.load(f)
            self.cat_alpha, self.cat_members = float(d["alpha"]), list(d["members"])
        else:
            self.cat_alpha, self.cat_members = 0.0, []
        self.cats = []

        self.fspec_alpha = float(bundle.get("fspec_alpha") or 0.0)"""


def main():
    t0 = time.time()
    from catboost import CatBoostClassifier, Pool
    import catboost
    print(f"catboost {catboost.__version__}  (제출 requirements 는 1.2.10)")

    art = joblib.load(T1 / "model" / "artifacts.joblib")["artifacts"]
    cols = lgb.Booster(model_str=(T1 / "model" / "lgbm_base_42.txt")
                       .read_text(encoding="utf-8")).feature_name()
    df = load_csv(TRAIN_CSV)
    X = t1_features(df, art, cols)
    w = recency_weight(df.season.to_numpy(), RECENCY_HALF_LIFE, ref_season=REF_SEASON)
    # ⛔ augment=False — 증강은 LGBM 전용이다 (헤더 참조)
    X2, y2, w2, init, k = build(df, X, w, False, np.random.default_rng(777))
    assert k == 0, f"증강이 걸렸다 (합성 {k}행) — CatBoost 에는 걸면 안 된다"
    print(f"[{time.time()-t0:.0f}s] 무증강 {len(X2):,}행 · 피처 {X2.shape[1]}  (합성 {k})")
    del X, df

    dst = ASCII / ARM
    (dst / "model").mkdir(parents=True, exist_ok=True)
    pool = Pool(X2, y2, weight=w2, baseline=init)
    del X2, y2, w2, init

    tags = []
    for var, over in CAT_VARIANTS.items():
        for sd in CAT_SEEDS:
            tag = f"{var}_{sd}"; tags.append(tag)
            f = dst / "model" / f"cat_{tag}.cbm"
            # 여기서는 **존재**가 유효한 재개 키다 — T1 은 cat_*.cbm 을 한 개도
            # 배포하지 않으므로 원본과 혼동될 수 없다. (lgbm 은 원본이 복사돼
            # 있어서 존재로 판단하면 안 됐다 — §7.176a 빌드에서 실제로 당했다.)
            if f.exists():
                print(f"[{time.time()-t0:.0f}s]   {tag:<14} 재사용")
                continue
            m = CatBoostClassifier(iterations=CAT_ITERS, random_seed=sd,
                                   **CAT_PARAMS, **over)
            m.fit(pool)
            m.save_model(str(f))
            print(f"[{time.time()-t0:.0f}s]   {tag:<14} 완료 ({f.stat().st_size/1e6:.1f} MB)")
            del m
    (dst / "model" / "cat_members.json").write_text(
        json.dumps({"alpha": CAT_ALPHA, "members": tags}), encoding="utf-8")
    print(f"  멤버 {len(tags)}개 · α={CAT_ALPHA}")

    # ── 패키징: 챔피언 zip + cat 파일 + predict.py 패치 + requirements ──
    with zipfile.ZipFile(BASE_ZIP) as z:
        names = z.namelist()
        # T1 계보의 predict.py 는 CRLF 다. LF 패턴이 안 맞으므로 정규화한 뒤
        # 패치한다 (파이썬 소스는 LF 로 저장해도 동작이 같다).
        pred = z.read("predict.py").decode("utf-8").replace(CRLF, LF)
        req = z.read("requirements.txt").decode("utf-8").replace(CRLF, LF)
    for old, new in ((INIT_SRC, INIT_NEW), (BLEND_SRC, BLEND_NEW)):
        assert pred.count(old) == 1, f"패치 대상 {pred.count(old)}회: {old[:40]!r}"
        pred = pred.replace(old, new)
    assert "import json" not in pred.split("class Predictor")[0], "json import 중복"
    pred = pred.replace("import os\nimport pickle", "import json\nimport os\nimport pickle")
    assert "catboost" not in req
    req = "catboost==1.2.10\n" + req

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(BASE_ZIP) as src, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for n in names:
            if n == "predict.py":
                out.writestr(n, pred)
            elif n == "requirements.txt":
                out.writestr(n, req)
            else:
                out.writestr(n, src.read(n))
        for tag in tags:
            out.write(dst / "model" / f"cat_{tag}.cbm", f"model/cat_{tag}.cbm")
        out.write(dst / "model" / "cat_members.json", "model/cat_members.json")
    assert len(OUT.name) <= 30
    print(f"\n{OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"(원본 {len(names)}개 + cat {len(tags)+1}개)")
    print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
