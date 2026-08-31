"""제출 zip 에 **컬럼 선형 δ 축**을 추가한다 (재학습 없음) — §7.223.

왜 새 도구인가
    `rebase_asof_axis.py` 가 다루는 `artifacts['asof']` 는 **조회형**이다
    (pitcher_id → 커리어 통계표). §7.223 이 살린 두 축은 그게 아니라
    **build_features 산출 컬럼 위의 선형항**이라 기존 통로가 없다.

        z += Σ_k β_k · (X[c_k] − μ_k) / σ_k          μ/σ 는 train fit

    그래서 이 도구는 아티팩트에 `art['col']` 리스트를 새로 만들고,
    `features.logit_adjust` 에 루프 하나 · `predict.py` 에 인자 하나를 더한다.
    (`logit_adjust(df, art)` → `logit_adjust(df, art, X=None)` — 기존 호출 호환)

    ⚠️ **`logit_adjust` 는 원래 `test_df`(원본 행)만 받는다.** `_sm`·`matchup_*` 는
    파생 컬럼이라 원본에 없다 → `predict.py` 가 이미 만들어 둔 `X` 를 넘겨야 한다.
    X 는 `X = X[self.cols]` 이후로 재대입되지 않는다(실측 확인).

■ μ/σ 는 **챔피언 자신의 파이프라인**에서 뽑는다
    우리 `src/features.py` 로 만든 컬럼과 챔피언 `features.py` 산출이 다르면
    β 눈금이 어긋난다 (§7.216 의 "원단위 β" 함정과 같은 자리). 그래서
    work 디렉터리의 `features.py` + `artifacts.joblib` 로 **전 시즌 train** 을
    청크로 흘려 μ/σ 를 잰다. 우리 코드 산출과의 대조는 G0 으로 매번 다시 한다.

■ 행 독립성
    행 자신의 컬럼 값 하나만 보는 선형항이다. `verify_zip.py` 의 L2(청크·셔플)
    관문이 이것을 다시 확인한다.

■ PROB_SHIFT 재보정
    `rebase_asof_axis.py` 와 같은 절차 — train 2024행 프록시로 Δmean 을 재고 뺀다.

사용
    python tools_hj/rebase_col_axis.py --src hj09_grbc.zip --out hj14_colax.zip \
        --axis asof_pitcher_middle_rate_sm=-0.0070 \
        --axis matchup_success_edge=0.0085
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np

from rebase_routing import SHIFT_RE, TRAIN_CSV, predict_in_subprocess, sha

CHUNK = 200_000

# ── features.py 패치 (CRLF 유지) ────────────────────────────────────────
SIG_OLD = "def logit_adjust(df, art):"
SIG_NEW = "def logit_adjust(df, art, X=None):"
TAIL_OLD = "    return z, hits"
COL_BLOCK = """    # 컬럼 선형 축 (§7.223). `X` 는 build_features 산출물이라 파생 컬럼
    # (`*_sm` · `matchup_*`)도 볼 수 있다. 행 자신의 값 하나만 보는 선형항이라
    # test 행 간 독립성에 영향이 없다.  z += beta * (x - mu) / sd
    for a in ((art.get("col") or []) if art else []):
        if float(a["beta"]) == 0.0:
            continue
        if X is None or a["name"] not in X.columns:
            raise ValueError("col 축 %s 를 X 에서 못 찾았다" % a["name"])
        x = np.asarray(X[a["name"]], dtype=np.float64)
        ok = np.isfinite(x)
        hits[a["name"]] = float(ok.mean())
        v = (np.where(ok, x, float(a["mu"])) - float(a["mu"])) / float(a["sd"])
        z += float(a["beta"]) * v
    return z, hits"""

CALL_OLD = "logit_adjust(test_df, self.art)"
CALL_NEW = "logit_adjust(test_df, self.art, X=X)"


def read_text(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def write_text(p, s):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def patch(p, pairs):
    s = read_text(p)
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{os.path.basename(p)}: 패턴 {n}개 (1개여야) — {old[:40]!r}"
        s = s.replace(old, new)
    write_text(p, s)


def champ_moments(work, cols, out_npz):
    """work 의 features.py + artifacts 로 전 시즌 train 을 흘려 μ/σ 를 잰다."""
    code = f"""
import sys, numpy as np, pandas as pd, joblib
sys.path.insert(0, {work!r})
from features import build_features
b = joblib.load({os.path.join(work, 'model', 'artifacts.joblib')!r})
art = b["artifacts"]
cols = {cols!r}
n = np.zeros(len(cols)); s1 = np.zeros(len(cols)); s2 = np.zeros(len(cols))
it = pd.read_csv({str(TRAIN_CSV)!r}, encoding="utf-8-sig", chunksize={CHUNK})
for ch in it:
    ch = ch.drop(columns=[c for c in ("control_success",) if c in ch.columns])
    X = build_features(ch.reset_index(drop=True), art)
    for i, c in enumerate(cols):
        v = np.asarray(X[c], dtype=np.float64)
        v = v[np.isfinite(v)]
        n[i] += v.size; s1[i] += v.sum(); s2[i] += (v * v).sum()
mu = s1 / n
sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
np.savez({out_npz!r}, mu=mu, sd=sd, n=n)
print("rows/col", n.astype(np.int64).tolist())
"""
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="4")
    r = subprocess.run([sys.executable, "-c", code], cwd=work, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[중단] μ/σ 계산 실패\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    print("  " + r.stdout.strip().splitlines()[-1])
    d = np.load(out_npz)
    return d["mu"], d["sd"]


def our_moments(cols):
    """G0 — 우리 `src/features.py` 로 같은 컬럼의 μ/σ 를 만든다 (대조용)."""
    import features as OF
    from config import (CONTAM_AXES, RECENCY_HALF_LIFE, SMOOTH_K_BATTER,
                        SMOOTH_K_PITCHER)
    from data import load_csv, recency_weight
    tr = load_csv(TRAIN_CSV)
    w = recency_weight(tr["season"].to_numpy(), RECENCY_HALF_LIFE,
                       ref_season=int(tr["season"].max()))
    art = OF.fit_artifacts(tr, weights=w, k_pitcher=SMOOTH_K_PITCHER,
                           k_batter=SMOOTH_K_BATTER, contam_axes=CONTAM_AXES)
    X = OF.build_features(tr, art)
    mu, sd = [], []
    for c in cols:
        v = np.asarray(X[c], dtype=np.float64)
        v = v[np.isfinite(v)]
        mu.append(float(v.mean())); sd.append(float(v.std()))
    return np.array(mu), np.array(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis", action="append", required=True,
                    help="컬럼명=β  (여러 번 지정 가능)")
    ap.add_argument("--skip-repro", action="store_true",
                    help="G0(우리 코드 대조) 생략 — 우리 build_features 는 느리다")
    a = ap.parse_args()

    import pandas as pd
    specs = []
    for s_ in a.axis:
        name, _, b = s_.partition("=")
        specs.append((name.strip(), float(b)))
    cols = [n for n, _ in specs]

    os.makedirs("D:/t1eval", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="colax_", dir="D:/t1eval")
    work, orig = os.path.join(tmp, "work"), os.path.join(tmp, "orig")
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist()
        z.extractall(work)
        z.extractall(orig)

    pp, fp = os.path.join(work, "predict.py"), os.path.join(work, "features.py")
    shift0 = float(SHIFT_RE.findall(read_text(pp))[0][1])
    ajp = os.path.join(work, "model", "artifacts.joblib")
    bundle = joblib.load(ajp)
    art = bundle["artifacts"]
    fcols = list(bundle["feature_cols"])
    print(f"소스 {os.path.basename(a.src)}   PROB_SHIFT {shift0:.9f}   피처 {len(fcols)}개")
    have = [x["name"] for x in (art.get("col") or [])]
    if have:
        print("  기존 col 축", [(x["name"], x["beta"]) for x in art["col"]],
              "— 여기에 추가한다")
    for c in cols:
        assert c in fcols, f"{c} 는 배포 피처 78개에 없다"
        assert c not in have, f"{c} 축이 이미 있다"
    print(f"  추가할 축 {specs}")

    # ── μ/σ (챔피언 파이프라인) ────────────────────────────────────────
    print("전 시즌 train 으로 μ/σ (챔피언 features.py) …")
    mu, sd = champ_moments(work, cols, os.path.join(tmp, "mom.npz"))
    for (c, b), m, s_ in zip(specs, mu, sd):
        print(f"  {c:<38} β {b:+.5f}  mu {m:+.9f}  sd {s_:.9f}")
    assert np.all(sd > 0), "sd = 0 인 축이 있다"

    if not a.skip_repro:
        print("G0 우리 코드 대조 …")
        omu, osd = our_moments(cols)
        for (c, _), m, s_, m2, s2 in zip(specs, mu, sd, omu, osd):
            rm = abs(m - m2) / max(abs(m), 1e-12)
            rs = abs(s_ - s2) / max(s_, 1e-12)
            print(f"  {c:<38} mu {m:+.9f}/{m2:+.9f} ({rm:.2%})  "
                  f"sd {s_:.9f}/{s2:.9f} ({rs:.2%})  "
                  f"{'OK' if max(rm, rs) < 0.01 else '⚠️ 1% 초과'}")

    # ── 코드 패치 ──────────────────────────────────────────────────────
    already = SIG_NEW in read_text(fp) and CALL_NEW in read_text(pp)
    if already:
        print("패치  col 통로가 이미 있다 — 코드 무변경 (축만 추가)")
    else:
        patch(fp, [(SIG_OLD, SIG_NEW), (TAIL_OLD, COL_BLOCK)])
        patch(pp, [(CALL_OLD, CALL_NEW)])
        print("패치  features.logit_adjust(+X, col 루프) · predict.logit_adjust(X=X)")

    # ── 3단 추론으로 PROB_SHIFT 재보정 ────────────────────────────────
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    df = df[df["season"] == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    pq = os.path.join(tmp, "proxy.pkl")
    df.to_pickle(pq)

    print("추론 ① 축 추가 전 (패치만) …")
    before = predict_in_subprocess(work, pq, os.path.join(tmp, "b.npy"))

    art["col"] = list(art.get("col") or []) + [
        {"name": c, "beta": float(b), "mu": float(m), "sd": float(s_)}
        for (c, b), m, s_ in zip(specs, mu, sd)]
    joblib.dump(bundle, ajp)

    print("추론 ② 축 추가 후 (레벨 미보정) …")
    mid = predict_in_subprocess(work, pq, os.path.join(tmp, "m.npy"))
    d_ = mid - before
    dm = float(d_.mean())
    print(f"  Δp 평균 {dm:+.6g} · 평균|Δp| {np.abs(d_).mean():.6g} · "
          f"max|Δp| {np.abs(d_).max():.6g}")
    assert np.abs(d_).max() > 0.0, "축이 예측에 반영되지 않았다"

    shift = shift0 - dm
    write_text(pp, SHIFT_RE.sub(lambda mm: f"{mm.group(1)}{shift:.9f}", read_text(pp)))
    print(f"  PROB_SHIFT {shift0:.9f} -> {shift:.9f}")

    print("추론 ③ 재보정 검증 …")
    after = predict_in_subprocess(work, pq, os.path.join(tmp, "a.npy"))
    r = float((after - before).mean())
    print(f"  재보정 후 Δmean {r:+.3g}  {'OK' if abs(r) < 1e-9 else '경고'}")
    assert abs(r) < 1e-7

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            f_ = os.path.join(work, n)
            if os.path.isfile(f_):
                z.write(f_, n)
    diff = sorted(n for n in names if os.path.isfile(os.path.join(work, n))
                  and open(os.path.join(work, n), "rb").read()
                  != open(os.path.join(orig, n), "rb").read())
    exp = sorted(["predict.py", "model/artifacts.joblib"]
                 + ([] if already else ["features.py"]))
    print(f"\n변경 격리  {len(diff)}/{len(names)}: {diff}")
    assert diff == exp, f"기대 {exp}"
    assert len(os.path.basename(a.out)) <= 30
    print(f"출력   {a.out}\n  SHA-256 {sha(a.out)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
