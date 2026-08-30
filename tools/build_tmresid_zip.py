"""트랙맨 잔차 층을 제출 zip 에 얹는다 (상대 팀 7층 이식).

기전
    챔피언 예측의 **잔차**를 트랙맨 투수 프로파일 + 문맥으로 학습한 LGBM 하나를
    최종 확률에 더한다.  `p ← clip(p + λ·resid_pred)`,  λ = 0.15.

    우리가 앞서 단변량 잔차 회귀로 0/20 · 0/13 을 얻고 "신호 없음" 이라 했던 것은
    **계측기가 약했던 것**이다. 단변량 t 검정은 상호작용을 원리적으로 못 본다.
    다변량 트리로 재니 위약 10회(평균 +4.16 · std 1.07 · 최대 +5.79) 대비
    실측 +7.01, **순효과 +2.84 · z 2.67** 로 위약 분포를 넘었다.

    ⚠️ λ 격자를 넓히지 않았으면 오판했다 — 0.25 가 이미 내리막이었고
       0.40 에서 −12.69, 1.0 에서 −219.96 이다. 꼭짓점은 0.15 다.

리키지
    `tm_profile.csv` 는 **학습 기간(≤2024) 트랙맨의 투수별 집계**다. 동결 상수표이고
    추론 시 그 행 자신의 `pitcher_id` 로만 조회한다. test 행 간 정보를 쓰지 않는다.
    트랙맨에 2025 는 없으며 매핑은 `control_success` 를 쓰지 않는다 (구조적 대응).
    규정 확인: DACON.GM #4(8/07) · #24(8/13) · #26(8/14) · #48(8/20).

    미지 투수(≈20%)는 조회 실패 → 결측. LGBM 이 결측을 그대로 다룬다.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

TRAIN_CSV = "/Users/hyejin/Aimers_9th/data/train.csv"
SHIFT_RE = re.compile(r"(PROB_SHIFT\s*=\s*)([0-9.eE+-]+)")

BLOCK = '''
        # ── 트랙맨 잔차 층 (팀 7층 이식) ─────────────────────
        # 동결 프로파일표(학습 기간 집계)를 그 행 자신의 pitcher_id 로 조회한다.
        # test 행 간 정보는 쓰지 않는다. 미지 투수는 결측 -> LGBM 이 처리.
        if self._tmr is not None:
            prof = self._tmprof.reindex(
                test_df["pitcher_id"].to_numpy()).reset_index(drop=True)
            ctx = pd.DataFrame({c: test_df[c].to_numpy() for c in self._tmctx})
            Xr = pd.concat([ctx, prof], axis=1)[self._tmctx + self._tmcols]
            p = np.clip(p + self._tmlam * self._tmr.predict(Xr), CLIP_EPS, 1 - CLIP_EPS)
'''


def predict_sub(w, pkl, out):
    code = ("import sys, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {w!r})\n"
            "from predict import Predictor\n"
            f"df = pd.read_pickle({pkl!r})\n"
            f"p = Predictor(model_dir={os.path.join(w,'model')!r})\n"
            f"np.save({out!r}, np.asarray(p.predict(df), dtype=np.float64))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=w, capture_output=True, text=True,
                       env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4"))
    if r.returncode:
        raise SystemExit(f"[중단] 추론 실패\n{r.stdout[-2000:]}\n{r.stderr[-2500:]}")
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    a = ap.parse_args()

    meta = json.load(open(a.meta, encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="tmr_")
    w, o = os.path.join(tmp, "w"), os.path.join(tmp, "o")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(w); z.extractall(o)

    shutil.copyfile(a.profile, os.path.join(w, "model", "tm_profile.csv"))
    shutil.copyfile(a.model, os.path.join(w, "model", "tm_resid.txt"))
    json.dump(meta, open(os.path.join(w, "model", "tm_resid_meta.json"), "w"), indent=2)

    pp = os.path.join(w, "predict.py")
    src0 = open(pp, encoding="utf-8").read()
    shift0 = float(SHIFT_RE.findall(src0)[0][1])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}   λ {meta['lam']}")

    # 적재 코드는 __init__ 끝에, 적용 코드는 return 직전에
    load = (f'\n        self._tmr = self._tmprof = None\n'
            f'        _tp = os.path.join(model_dir, "tm_profile.csv")\n'
            f'        _tm = os.path.join(model_dir, "tm_resid.txt")\n'
            f'        if os.path.exists(_tp) and os.path.exists(_tm):\n'
            f'            import pandas as _pd\n'
            f'            self._tmprof = _pd.read_csv(_tp).set_index("pitcher_id")\n'
            f'            self._tmr = lgb.Booster(model_file=_tm)\n'
            f'            self._tmctx = {meta["ctx"]!r}\n'
            f'            self._tmcols = list(self._tmprof.columns)\n'
            f'            self._tmlam = {meta["lam"]!r}\n')
    anchor = "\n    def predict(self, test_df):"
    assert anchor in src0, "predict 정의를 못 찾았다"
    s = src0.replace(anchor, load + anchor, 1)
    ret = "        return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)"
    assert ret in s, "return 문을 못 찾았다"
    s = s.replace(ret, BLOCK + ret, 1)
    if "import pandas as pd" not in s:
        s = s.replace("import numpy as np", "import numpy as np\nimport pandas as pd", 1)
    open(pp, "w", encoding="utf-8").write(s)

    proxy = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    proxy = proxy[proxy.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pk = os.path.join(tmp, "p.pkl"); proxy.to_pickle(pk)

    print("추론 ① 잔차층 전 …")
    before = predict_sub(o, pk, os.path.join(tmp, "b.npy"))
    print("추론 ② 잔차층 후 (레벨 미보정) …")
    mid = predict_sub(w, pk, os.path.join(tmp, "m.npy"))
    d = mid - before; dm = float(d.mean())
    print(f"  max|Δp| {np.abs(d).max():.6g} · 평균|Δp| {np.abs(d).mean():.6g} · Δmean {dm:+.6g}")
    shift = shift0 - dm
    open(pp, "w", encoding="utf-8").write(SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", s))
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")
    print("추론 ③ 재보정 검증 …")
    after = predict_sub(w, pk, os.path.join(tmp, "a.npy"))
    print(f"  패치 후 Δmean {float((after-before).mean()):+.3g}")

    alln = names + ["model/tm_profile.csv", "model/tm_resid.txt", "model/tm_resid_meta.json"]
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in alln:
            fp = os.path.join(w, n)
            if os.path.isfile(fp):
                z.write(fp, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(w, n))
                  and open(os.path.join(w, n), "rb").read() != open(os.path.join(o, n), "rb").read())
    print(f"\n기존 파일 중 변경 {diff}  · 신규 3개 추가")
    if diff != ["predict.py"]:
        raise SystemExit("[중단] predict.py 외 변경 감지")
    print(f"출력 {a.out}\n  SHA-256 {hashlib.sha256(open(a.out,'rb').read()).hexdigest()}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
