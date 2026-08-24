"""제출 zip 빌드 + 리키지 정적 검사 + 추론 시간 측정.

실행:  .venv/Scripts/python.exe -u src/build_submit.py
산출:  submit.zip  (model/, script.py, features.py, predict.py, requirements.txt)
"""
import ast
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# 콘솔 인코딩이 UTF-8이 아니면(예: Windows CP949) print 한 줄에 프로세스가 죽는다.
# 평가 서버의 로케일을 알 수 없으므로 방어적으로 고정한다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd

from config import CAT_ALPHA, ROOT, TEST_CSV, TRAIN_CSV
from predict import (APN_EDGES, APN_SHIFT, CTX_BETA_MULT, FSPEC_ALPHA_OVERRIDE,
                     MLP_ALPHA_OVERRIDE, PROB_CENTER,
                     PROB_SCALE, PROB_SHIFT, PROB_TARGET, SEG_SHIFT)

SRC = ROOT / "src"
SUBMIT = ROOT / "submit"

# src → submit 으로 복사되는 파일 (단일 진실 소스는 항상 src/)
# mlp_infer.py는 **numpy 전용** 순전파다 — torch는 제출물에 들어가지 않는다.
COPY_FILES = ["features.py", "predict.py", "mlp_infer.py", "contam.py"]

# SUBMISSIONS.md **제출 표**의 행. 본문의 다른 표(fold·α 스윕 등)와 구분하려면
# "번호 + 백틱으로 시작하는 파일명"을 둘 다 요구해야 한다 (§s26에서 속았다).
# `re.M` 은 파일 전체를 finditer 할 때 필요하고, 한 줄 match 에는 무해하다.
ROW_RE = re.compile(r"^\| (\d+) \| `", re.M)

# 추론 코드에서 test 데이터프레임에 대해 호출되면 안 되는 것들 (CLAUDE.md §3.2)
FORBIDDEN_CALLS = {
    "groupby", "value_counts", "rolling", "expanding", "cumsum", "cumcount",
    "fit", "fit_transform", "nlargest", "nsmallest", "rank", "duplicated",
}
# 분포 통계 함수도 금지 (test 전체를 보고 계산하는 값)
FORBIDDEN_ATTRS = {"quantile", "describe", "corr", "cov", "mode"}

REQUIREMENTS = """\
lightgbm==4.6.0
catboost==1.2.10
numpy==2.2.6
pandas==2.3.3
scipy==1.15.2
joblib==1.4.2
scikit-learn==1.6.1
"""


def leakage_scan(paths):
    """AST로 금지 호출을 찾는다. 문자열 grep이 아니라 실제 호출 노드를 본다."""
    hits = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS | FORBIDDEN_ATTRS:
                    hits.append((p.name, node.lineno, node.func.attr))
    return hits


def make_dummy_test(n=245789, art=None):
    """평가 규모(245,789행)의 더미 test.

    2024 시즌 행에 `season = 2025` 라벨을 붙인다.

    ⚠️ **`art`를 주지 않으면 예측 평균을 읽으면 안 된다** (§7.121).
       `asof_*`는 **커리어 누적**이고 시즌 진행분은 `n_szn = asof_n − n_off(id, season)`
       으로 계산되는데, `n_off(·, 2025)`는 **2024 종료 시점 누적**이다. 2024 행의
       `asof_n`은 항상 그보다 작으므로 라벨만 바꾸면 **100% 행이 `n_szn ≤ 0`**
       (평균 −708.7)이 되어 축 2개가 상수로 죽는다. 실제 2025 test는 `n_szn > 0`이다.
       그 상태로 잰 평균은 **+0.0044 위로 편향**된다 (0.4758 → 0.4801).

       `art`(= artifacts.joblib의 "artifacts")를 주면 시즌 진행분을 보존해
       **재기준화**한다: `n' = n_off₂₅ + (n − n_off₂₄)`, `k`도 같은 방식.
       추론 **시간** 측정만 할 거면 `art` 없이 써도 무해하다.
    """
    df = pd.read_csv(TRAIN_CSV)
    df = df[df.season == 2024].drop(columns=["control_success"]).reset_index(drop=True)
    reps = int(np.ceil(n / len(df)))
    out = pd.concat([df] * reps, ignore_index=True).iloc[:n].copy()
    out["season"] = 2025
    out["row_id"] = [f"TEST_{i:06d}" for i in range(n)]

    for a in (art or {}).get("asof", []):
        ids = out[a["id"]].to_numpy(dtype=np.int64)

        def _off(season):
            kv = ids * 10000 + season
            j = np.clip(np.searchsorted(a["keys"], kv), 0, max(len(a["keys"]) - 1, 0))
            hit = (len(a["keys"]) > 0) & (a["keys"][j] == kv)
            return np.where(hit, a["n_off"][j], 0.0), np.where(hit, a["k_off"][j], 0.0)

        n24, k24 = _off(2024)
        n25, k25 = _off(2025)
        n_old = out[a["n_col"]].to_numpy(dtype=np.float64)
        # r×n 의 정수 복원은 asof_lookup 이 쓰는 전제와 같다 (max|r×n − round| < 0.008)
        k_old = np.round(out[a["r_col"]].to_numpy(dtype=np.float64) * n_old)
        n_new = n25 + (n_old - n24)
        k_new = k25 + (k_old - k24)
        out[a["n_col"]] = n_new
        out[a["r_col"]] = np.where(n_new > 0, k_new / np.maximum(n_new, 1.0),
                                   float(a["league"]))
    return out


def main():
    print("=== 1. 리키지 정적 검사 ===")
    targets = [SUBMIT / "script.py"] + [SRC / f for f in COPY_FILES]
    hits = leakage_scan(targets)
    if hits:
        for f, ln, call in hits:
            print(f"  ❌ {f}:{ln}  .{call}()")
        raise SystemExit("추론 경로에 금지 호출이 있다. 제출 중단.")
    print("  ✅ groupby/value_counts/rolling/fit/quantile 등 금지 호출 없음")

    print("\n=== 2. src → submit 복사 ===")
    for f in COPY_FILES:
        shutil.copy2(SRC / f, SUBMIT / f)
        print(f"  {f}")
    (SUBMIT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    model_dir = SUBMIT / "model"
    for f in ["artifacts.joblib", "meta.json"]:
        if not (model_dir / f).exists():
            raise SystemExit(f"model/{f} 없음. 먼저 src/train.py를 실행할 것.")
    boosters = sorted(model_dir.glob("lgbm_*.txt"))
    if not boosters:
        raise SystemExit("model/lgbm_*.txt 없음. 먼저 src/train.py를 실행할 것.")
    print(f"  부스터 {len(boosters)}개: {[p.name for p in boosters]}")
    cats = sorted(model_dir.glob("cat_*.cbm"))
    _bcat = joblib.load(model_dir / "artifacts.joblib").get("cat_members") or []
    if _bcat and len(cats) != len(_bcat):
        raise SystemExit(f"CatBoost 멤버 불일치 — 디스크 {len(cats)}개 vs "
                         f"번들 {len(_bcat)}개. model/ 정리 필요.")
    if cats:
        print(f"  CatBoost {len(cats)}개 (alpha={CAT_ALPHA})")
    elif CAT_ALPHA > 0:
        raise SystemExit("CAT_ALPHA > 0 인데 model/cat_*.cbm 없음. train.py 재실행.")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    # 확률공간 레벨 시프트는 **추론 층 상수**라 train.py가 쓴 meta.json에 없다.
    # 태그·지문·이력이 이 축을 보게 하려면 여기서 얹어야 한다 (describe_tokens 참조).
    meta["prob_shift"] = PROB_SHIFT
    meta["prob_scale"] = PROB_SCALE
    meta["seg_shift"] = SEG_SHIFT
    meta["apn_shift"] = list(APN_SHIFT) if APN_SHIFT else None
    meta["apn_edges"] = list(APN_EDGES) if APN_SHIFT else None
    meta["ctx_beta_mult"] = CTX_BETA_MULT
    if FSPEC_ALPHA_OVERRIDE is not None:
        meta["fspec_alpha"] = FSPEC_ALPHA_OVERRIDE   # 배포층 오버라이드가 이긴다
    if MLP_ALPHA_OVERRIDE is not None:
        meta["mlp_alpha"] = MLP_ALPHA_OVERRIDE
    # ⚠️ `mlp_variant_weight`는 train.py가 **config 값**을 적어둔 것이다.
    #    artifacts.joblib의 `mlp_weights`를 직접 패치하면(§7.120 `_vw` 축)
    #    meta.json은 옛 값을 그대로 들고 있어 태그가 축 변경을 놓친다 —
    #    §7.61이 정확히 이 실패였다. **서빙되는 값**을 진실 소스로 쓴다.
    _b = joblib.load(model_dir / "artifacts.joblib")
    _mem, _w = _b.get("mlp_members") or [], _b.get("mlp_weights") or []
    if _mem and _w:
        served = {}
        for m, wt in zip(_mem, _w):
            served[m.split("_")[0]] = served.get(m.split("_")[0], 0.0) + float(wt)
        cfg = meta.get("mlp_variant_weight") or {}
        if cfg and any(abs(served.get(g, 0.0) - float(v)) > 1e-9 for g, v in cfg.items()):
            print(f"  ⚠️ mlp_variant_weight 불일치 — config {cfg} vs 서빙 {served}")
            print("     (artifacts.joblib 직접 패치. 태그는 서빙값 기준으로 붙인다)")
        meta["mlp_variant_weight"] = served
    # 같은 이유로 `fspec_alpha` 도 **서빙되는 값**에서 읽는다 (§7.132).
    # s35 첫 빌드가 아티팩트를 0.4580 으로 패치했는데 meta.json 이 0.15 라
    # 태그가 `fs15` 로 나왔다 — **파일명이 거짓말을 했다.**
    if _b.get("fspec_alpha") is not None:
        cfa, sfa = meta.get("fspec_alpha"), float(_b["fspec_alpha"])
        if cfa is not None and abs(float(cfa) - sfa) > 1e-12:
            print(f"  ⚠️ fspec_alpha 불일치 — meta {cfa} vs 서빙 {sfa}  (서빙값으로 태그)")
        meta["fspec_alpha"] = sfa
    # CatBoost 도 같은 이유로 **서빙되는 값**에서 읽는다 (§7.132 태그 거짓말 전례).
    if _b.get("cat_alpha") is not None:
        cca, sca = meta.get("cat_alpha"), float(_b["cat_alpha"])
        if cca is not None and abs(float(cca) - sca) > 1e-9:
            print(f"  ⚠️ cat_alpha 불일치 — meta {cca} vs 서빙 {sca}  (서빙값으로 태그)")
        meta["cat_alpha"] = sca
        meta["cat_members"] = list(_b.get("cat_members") or [])

    mlps = sorted(model_dir.glob("mlp_*.npz"))
    if meta.get("use_mlp"):
        if not mlps:
            raise SystemExit("USE_MLP인데 model/mlp_*.npz 없음. src/train.py 재실행.")
        print(f"  MLP {len(mlps)}개: {[p.name for p in mlps]}  "
              f"(alpha={meta['mlp_alpha']})")
    elif mlps:
        raise SystemExit("USE_MLP=False인데 mlp_*.npz가 남아 있다. model/ 정리 필요.")

    print("\n=== 3. 추론 시간 측정 (245,789행 더미) ===")
    dummy_dir = SUBMIT / "data"
    dummy_dir.mkdir(exist_ok=True)
    # 재기준화에 asof 오프셋이 필요하다 (§7.121). `_b`는 위에서 이미 읽었다.
    dummy = make_dummy_test(art=_b.get("artifacts"))
    print("  더미: 시즌 진행분 재기준화 적용 (asof n_szn > 0 유지)")
    dummy.to_csv(dummy_dir / "test.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": dummy["row_id"], "control_success": 0.5}).to_csv(
        dummy_dir / "sample_submission.csv", index=False, encoding="utf-8")

    t0 = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=SUBMIT,
                       capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t0
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("script.py 실행 실패")
    print(f"  ⏱ {elapsed:.1f}s / 600s 한도  ({elapsed/600*100:.1f}%)")
    if elapsed > 480:
        print("  ⚠️ 여유가 20% 미만이다. 트리 수를 줄이는 것을 검토할 것.")

    out = pd.read_csv(SUBMIT / "output" / "submission.csv")
    assert list(out.columns) == ["row_id", "control_success"], out.columns
    assert out["control_success"].between(0, 1).all()
    assert out["control_success"].notna().all()
    print(f"  submission 검증 OK  mean={out['control_success'].mean():.4f}")
    # 레벨 시프트가 켜져 있으면 **도달 평균**을 대조한다. PROB_SHIFT는 특정
    # 아티팩트에서 잰 p̄에 묶인 상수라, 재학습하면 조용히 낡는다 — 그 경우
    # 레벨 항이 어긋난 채로 제출된다 (§7.120: 어긋남 0.005가 −10 BSS 규모).
    # ⚠️ **APN 보정분을 정확히 되돌린 뒤에 대조한다** (§7.152-④ 후속).
    #
    # `make_dummy_test` 는 `n' = n_off₂₅ + (n − n_off₂₄)` 로 **전원을 한 살씩**
    # 늙힌다 — 신규 유입(로스터 회전)이 없다. 그래서 `asof_pitcher_n` 버킷 점유율이
    # 실제와 크게 다르다. 검증 가능한 유일한 전환(2023→2024)에 같은 변환을 걸어
    # 실측했다 (`src/occval.py`):
    #
    #     최저 버킷   더미 예측 0.0172   실제 0.1075   → **6.2배 빗나간다**
    #     최고 버킷   더미 예측 0.6702   실제 0.4511
    #
    # 실측 점유율은 2022 이후 안정적이다 (.1092 → .0907 → .1075). §s14 가 기록한
    # "감쇠 모형에 개체 회전을 넣어라"의 재현이고, 그때도 3배 빗나갔다.
    # → **더미의 버킷 분포에 레벨 상수를 맞추면 안 된다.** APN 항을 정확히 역산해
    #   빼고, 남은 것을 기존 검사에 그대로 통과시킨다 (검사의 보호 성질 보존:
    #   재학습으로 p̄ 가 움직이면 여전히 잡힌다).
    if PROB_SHIFT:
        got_raw = float(out["control_success"].mean())
        pv = out["control_success"].to_numpy(dtype=np.float64)
        if APN_SHIFT:
            gg = np.digitize(dummy["asof_pitcher_n"].to_numpy(dtype=np.float64),
                             np.asarray(APN_EDGES, dtype=np.float64))
            add = np.asarray(APN_SHIFT, dtype=np.float64)[gg]
            q = np.clip(pv - PROB_SHIFT, 1e-9, 1 - 1e-9)      # 아핀 직후 확률
            q0 = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) - add)))   # APN 역산
            pv = q0 + PROB_SHIFT
            occ = np.bincount(gg, minlength=len(APN_SHIFT)) / len(gg)
            print(f"  APN 더미 점유율 {' '.join(f'{x:.4f}' for x in occ)}"
                  f"  (⚠️ 회전 없음 — 실제와 다르다)")
            print(f"  APN 이 더미 평균에 준 이동 {got_raw - float(pv.mean()):+.6f}"
                  f"  → 역산해 제외하고 대조한다")
        got = float(pv.mean())
        off = got - PROB_TARGET
        print(f"  레벨 대조(재기준화 더미)  실측 {got:.6f}  목표 {PROB_TARGET:.6f}  어긋남 {off:+.6f}")
        if abs(off) > 5e-4:
            print(f"  ❌ 레벨이 목표에서 {off:+.6f} 벗어났다. PROB_SHIFT가 낡았다.")
            print(f"     현 아티팩트에서 p̄를 다시 재고 "
                  f"PROB_SHIFT = {PROB_TARGET} − p̄ 로 갱신할 것.")
            raise SystemExit("레벨 대조 실패. 제출 중단.")
    if meta.get("trackman_mode", "none") != "none":
        print("  ⚠️ 위 평균을 목표 prior와 비교하지 말 것. 더미는 2024 행이라 Trackman")
        print("     조회 적중률이 ~99%로 실제 2025(~81%)보다 높고, 커버리지가 높을수록")
        print("     평균이 낮게 나온다. 실제 목표 달성 여부는 train.py의 참조 분포")
        print("     (신인 비율까지 재현) 기준으로 이미 보정돼 있다.")

    print("\n=== 4. zip 패키징 ===")
    shutil.rmtree(dummy_dir, ignore_errors=True)
    shutil.rmtree(SUBMIT / "output", ignore_errors=True)

    # 이 설정에서 실제로 쓰지 않는 아티팩트는 넣지 않는다.
    # (남아 있으면 코드 검증자가 "왜 안 쓰는 테이블이 있나" 하고 의심한다)
    skip = set()
    if meta.get("trackman_mode", "none") == "none":
        skip |= {"trackman_pitcher.csv", "pitcher_id_map.csv"}

    # ⚠️ zip 은 `submit/` 아래를 **무조건 재귀로** 담는다. 백업 모델 디렉토리를
    #    여기 두면 통째로 들어가서 (실제로 s32 빌드가 62.7MB 가 됐다) 코드 검증자가
    #    "왜 안 쓰는 모델이 셋인가" 하고 본다 — 위 `skip` 주석과 같은 이유다.
    allow = {"model", "script.py", "requirements.txt", "history.json", *COPY_FILES}
    extra = sorted(x.name for x in SUBMIT.iterdir()
                   if x.name not in allow and x.name != "__pycache__")
    if extra:
        for x in extra:
            print(f"  ❌ submit/ 최상위에 예상 밖 항목: {x}")
        raise SystemExit("submit/ 밖으로 옮기고 다시 실행할 것.")

    zip_path = next_zip_path(meta)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SUBMIT.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name not in skip:
                z.write(p, p.relative_to(SUBMIT).as_posix())
    if skip:
        print(f"  제외: {sorted(skip)}  (TRACKMAN_MODE=none)")
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            print(f"  {i.file_size:>10,}  {i.filename}")
    print(f"\n✅ {zip_path.name}  ({zip_path.stat().st_size/1e6:.1f} MB)")

    record_history(zip_path, meta)
    log_submission(zip_path, meta, float(out["control_success"].mean()), elapsed)


MAX_NAME = 30      # 제출 파일명 길이 제한


def describe_tokens(meta):
    """제출물을 구분하는 짧은 태그. 30자 제한이 있으므로 최대한 압축한다.
    전체 설정은 SUBMISSIONS.md에 있고, 파일명은 서로 구분만 되면 된다.

    예: `s09_p475_h5_f80_e20m8a40.zip`
        prior 0.475 / half_life 0.5 / 피처 80개 / LGBM 20 + MLP 8 / alpha 0.40

    ⚠️ **블렌드 가중 alpha를 반드시 넣는다.** 이게 빠져 있어서 s08(a=0.30)과
       s09(a=0.40)의 태그가 번호만 빼고 글자 단위로 같아진 적이 있다.
       이름으로 설정을 구분하는 게 이 함수의 존재 이유다 —
       **새 하이퍼파라미터를 배포에 넣으면 여기에도 넣어야 한다.**
    """
    #   `p475`(prior)   s13에서 뺌. s01 이후 12회 불변이고 표에 'prior' 열로 있다
    parts = []
    # 자리를 비운 것들 — 전부 **여러 제출에 걸쳐 안 바뀌었고** SUBMISSIONS.md에 열로 있다.
    #   `f80`(피처 수)   s11에서 뺌. s05 이후 불변
    #   `h5`(recency)    s12에서 뺌. s04 이후 불변이고 §7.23에서 재검증·기각됐다
    #   `m8`(MLP 멤버 수) s17에서 뺌. s09 이후 8회 불변이고 `a40`이 MLP 사용을 이미 알린다
    #   `e20`(LGBM 멤버)  s18에서 뺌. s13 이후 5회 불변이고 meta.json에 n_members로 있다
    # 30자 제한에 걸리면 **이번에 바뀐 축이 아니라 안 바뀐 축을 뺀다.**
    #
    # ⚠️ 축이 8개가 되면서 26자 예산이 구조적으로 부족해졌다. 지금은 뺄 수 있는
    #    불변 축이 남아 있지만 **다음 축에서는 이 방식이 한계에 닿는다.**
    #    그때는 태그를 늘리지 말고 `sNN` + meta.json 해시 4자리 같은 방식으로 바꾼다
    #    — 파일명은 서로 구분만 되면 되고, 설정 전문은 SUBMISSIONS.md와 meta.json에 있다.
    tok = {}
    if meta.get("use_mlp"):
        tok["alpha"] = "a" + f"{float(meta.get('mlp_alpha', 0)):.2f}".split(".")[1]
        # noid 비중 (MODEL.md §7.27). 균등(.375)이면 생략 — 옛 태그와 호환된다.
        nw = float((meta.get("mlp_variant_weight") or {}).get("noid", 0.0))
        tw = sum(float(v) for v in (meta.get("mlp_variant_weight") or {}).values())
        if tw > 0 and abs(nw / tw - 0.375) > 1e-6:
            tok["noid"] = "n" + f"{nw / tw * 10:.0f}"   # 0.625 -> n6
    # 사후 로짓 보정 축의 β (MODEL.md §7.36 hand · §7.37 count4).
    # 축이 늘면 태그도 늘어난다 — 축마다 접두 1~2자로 붙인다. 없으면 생략(옛 태그 호환).
    #   `pl15`(hand β)   s15에서 뺌. s11 이후 4회 불변이고 meta.json에 ctx_axes로 있다
    #   `c10`(count4 β)  s17에서 뺌. s12 이후 5회 불변. 같은 이유·같은 대체 수단이다.
    # ⚠️ MUTE에 넣는 것은 "값이 안 바뀐다"는 뜻이 아니라 "지금까지 안 바뀌었다"는 뜻이다.
    #    이 축을 다시 건드리는 실험을 하면 **여기서 빼야** 태그가 두 제출을 구분한다.
    # ✅ s20에서 **MUTE를 비웠다.** §7.61이 두 β를 바꿨기 때문이다 (1.5→1.875,
    #    1.0→1.25). 위 경고가 예고한 상황이 실제로 왔고, 처음 빌드에서 태그가
    #    s19와 번호만 다르게 나왔다 — 경고를 써놓고 안 지킨 것이다.
    #    인코딩을 ×10 → **×100**으로 올렸다. 1.875는 ×10이면 "19"로 뭉개진다.
    #    (옛 태그 pl15·c10과는 자릿수가 달라 헷갈릴 수 있으나, 그건 이미 디스크에
    #     고정된 파일명이고 describe()는 새 이름만 만든다.)
    #
    # CTX_HALF_LIFE는 태그에 없다 — 가중을 바꾸면 δ의 스케일이 바뀌어 **β 재선정이
    # 필수**이므로(고원이 이동한다) β가 항상 같이 움직인다. 정확한 값은 meta.json의
    # `ctx_half_life`에 있다.
    ABBR = {"hand": "h", "count4": "c"}          # 1.875 -> h188,  1.25 -> c125
    for name, beta in (meta.get("ctx_axes") or {}).items():
        if beta:
            tok["ctx:" + name] = ABBR.get(name, name[:2]) + f"{float(beta) * 100:.0f}"
    # 다중 과제 보조 헤드 λ (MODEL.md §7.41). 0이면 생략 — 옛 태그와 호환된다.
    #   `x10`(λ)  s18에서 뺌. s13 이후 5회 불변이고 meta.json에 mlp_aux_lambda로 있다.
    #             ⚠️ 헤드 구성을 다시 실험하면 이 블록을 되살려야 한다 (§7.42).
    ax = 0.0 if meta.get("mlp_aux_lambda") == 1.0 else float(meta.get("mlp_aux_lambda") or 0.0)
    if ax:
        t = "x" + f"{ax * 10:.0f}"
        # 축 개수는 **2가 아닐 때만** 붙인다 → s13의 `x10`이 그대로 유지된다.
        # λ만으로는 헤드 2개짜리와 4개짜리가 같은 태그를 받는다 (§7.42).
        nax = len(meta.get("mlp_aux_modes") or [])
        if nax and nax != 2:
            t += f"h{nax}"
        tok["aux"] = t
    # game_type 외 추가 드랍 피처 (MODEL.md §7.78 달력 제거). 기본(없음)이면 생략.
    #   game_month+game_dayofweek -> nocal.  다른 조합이 오면 개수로 구분한다.
    extra_drop = sorted(c for c in (meta.get("drop_features") or [])
                        if c != "game_type")
    if extra_drop:
        tok["dropx"] = ("nocal" if extra_drop == ["game_dayofweek", "game_month"]
                        else f"dx{len(extra_drop)}")
    # pre-2023 F 오염 보정 축 (MODEL.md §7.49). 끄면 생략 — 옛 태그와 호환된다.
    #   cfp = 투수 축만,  cfpb = 투수+타자
    #   `cfpb`  s20에서 뺌. s14 이후 6회 불변이고 meta.json에 contam_axes로 있다.
    #           ctx β 두 개를 되살리느라 8자가 필요해서 자리를 만들었다.
    ca = meta.get("contam_axes") or ()
    if ca and set(ca) != {"p", "b"}:
        tok["contam"] = "cf" + "".join(ca)
    # init_score 로 (시즌 × game_type) 레벨 제거 (MODEL.md §7.109). 끄면 생략.
    # ⚠️ **이 함수가 세 번째로 같은 일을 당했다.** 새 축을 배포에 넣으면서 식별 함수에
    #    안 넣으면 빌드가 직전 제출과 같은 이름을 만든다 (s21의 asof_strength,
    #    그리고 이번 §7.109). foldcache._key·jointbeta pin 과 같은 부류다 —
    #    **식별자를 만드는 함수는 새 노브를 자동으로 알지 못한다.**
    if meta.get("use_init_score"):
        tok["init"] = "is"
    # 시즌 진행분 (MODEL.md §7.55 사후가산 · §7.56 컬럼). 둘은 배타적이다.
    #   사후 가산  0.02 -> as2   (β는 δ축보다 두 자릿수 작으므로 ×100)
    #   컬럼       gap_success+n_szn -> ac2   (ac + 컬럼 수)
    # 끄면 생략 — 옛 태그와 호환된다.
    AB = {"gap_success": "as", "gap_bsuccess": "bs"}
    for name, beta in (meta.get("asof_axes") or {}).items():
        if beta:
            tok["asof:" + name] = AB.get(name, name[:2]) + f"{float(beta) * 1000:.0f}"
    # EB 수축 강도 (MODEL.md §7.63). 200은 한 번도 스윕 안 된 채 배포돼 있었다.
    # ⚠️ **배포 하이퍼파라미터인데 태그에 없었다.** s21에서 `next_zip_path`의 새 assert가
    #    잡았다 — 이번엔 β_b도 같이 바뀌어 우연히 구분됐지만, strength만 바꿨으면
    #    태그가 직전과 같았을 것이다. 기본값(200)일 때만 생략해 옛 태그와 호환한다.
    st = float(meta.get("asof_strength") or 0.0)
    if (meta.get("asof_axes") or {}) and st and st != 200.0:
        tok["k"] = f"k{st:.0f}"
    # 레퍼토리 균형도 (MODEL.md §7.58).  r3_ent 0.02 -> me20  (asof와 같은 ×1000)
    MB = {"r3_ent": "me"}
    for name, beta in (meta.get("mix_axes") or {}).items():
        if beta:
            tok["mix:" + name] = MB.get(name, name[:2]) + f"{float(beta) * 1000:.0f}"
    # 행 단위 F 레벨 δ_F (MODEL.md §7.69).  −0.0711×0.5 → gf36 (×1000, 부호는
    # 항상 음수라 생략 — 부호까지 구분은 지문이 한다). 끄면 생략.
    gtd = float(meta.get("gt_f_delta") or 0.0) * float(meta.get("gt_f_beta") or 0.0)
    if gtd:
        tok["gt"] = f"gf{abs(gtd) * 1000:.0f}"
    # 2019–22 F 오염 라벨 soft label (MODEL.md §7.70). 끄면 생략 — 옛 태그와 호환.
    # h_s는 학습 데이터에서 유도되는 값(자유 파라미터 아님)이라 on/off만 표기한다.
    if meta.get("soft_label_f"):
        tok["soft"] = "sl"
    # F 전용 전문가 블렌드 (MODEL.md §7.101).  α=0.15 -> fs15.  끄면 생략.
    # ⚠️ α만 태그한다. 학습 시즌 하한(FSPEC_MIN_SEASON)은 meta.json에 있고,
    #    그 값을 건드리는 실험을 하면 여기에 같이 넣어야 한다 (§7.63의 교훈).
    fa = float(meta.get("fspec_alpha") or 0.0)
    if fa:
        tok["fspec"] = f"fs{fa * 100:.0f}"
    # CatBoost 앙상블 (MODEL.md §7.140~§7.145).  α=0.10 · 12멤버 -> cb10m12.
    # ⚠️ 멤버 수까지 넣는다 — α 를 고정한 채 변형/시드를 바꾸는 실험이
    #    다음 후보로 잡혀 있어서(R-only 라우팅·변형 재선정), α 만 태그하면
    #    §7.63·§7.132 가 두 번 당한 "태그가 두 제출을 구분 못 함"이 재현된다.
    ca = float(meta.get("cat_alpha") or 0.0)
    if ca:
        tok["cat"] = f"cb{ca * 100:.0f}m{len(meta.get('cat_members') or [])}"
    # δ β 공통배율 (MODEL.md §7.131).  1.10 -> bm110.  1.0 이면 생략.
    bm = float(meta.get("ctx_beta_mult") or 1.0)
    if bm != 1.0:
        tok["bmult"] = f"bm{bm*100:.0f}"
    # MLP 멤버 수 (MODEL.md §7.128).  s17 에서 "s09 이후 8회 불변"이라 뺐는데
    # §7.128 이 8 -> 16 으로 바꿨다. **기본값 8 이 아니면 태그한다.**
    nm = len(meta.get("mlp_members") or [])
    if meta.get("use_mlp") and nm and nm != 8:
        tok["nmlp"] = f"m{nm}"
    # 세그먼트 레벨 분해 (MODEL.md §7.123).  D̂ = Δ_R − Δ_F, x1e5 -> sg841.
    # 끄면(=None/빈dict) 생략 — 옛 태그와 호환된다.
    sg = meta.get("seg_shift") or {}
    if sg:
        tok["seg"] = f"sg{abs(sg['R'] - sg['F']) * 1e5:.0f}"
    # 커리어 단계별 레벨 보정 (MODEL.md §7.151).  |a_1|×1e5 -> ap1984.  끄면 생략.
    # ⚠️ **κ 가 아니라 계수 자체를 태그한다.** κ 만 넣으면 같은 κ 로 a_g 를 재적합한
    #    제출(스택이 바뀌면 반드시 그렇게 된다)이 직전과 같은 이름을 받는다 —
    #    §7.63·§7.132·§7.146 이 세 번 당한 그 실패다.
    ap = meta.get("apn_shift") or []
    if ap:
        tok["apn"] = f"ap{abs(float(ap[0])) * 1e5:.0f}"
    # 확률공간 스케일 c (MODEL.md §7.120).  1.05 -> c105.  1.0이면 생략.
    pc = float(meta.get("prob_scale") or 1.0)
    if pc != 1.0:
        tok["pscale"] = f"c{pc * 100:.0f}"
    # 확률공간 레벨 시프트 (MODEL.md §7.120).  +0.006874 -> lv687  (x1e5, 반올림)
    # 음수면 `lvm...`. 0이면 생략 — 옛 태그와 호환된다.
    ps = float(meta.get("prob_shift") or 0.0)
    if ps:
        tok["lv"] = f"lv{'' if ps > 0 else 'm'}{abs(ps) * 1e5:.0f}"
    tm = meta.get("trackman_mode", "none")
    if tm != "none":                        # 기본값(끔)일 때는 태그를 생략한다
        tok["tm"] = {"release": "tmR", "release_ext": "tmE"}.get(tm, "tmA")
    return tok


def describe(meta):
    """전체 설정 태그 (SUBMISSIONS.md 기록용). **파일명에는 더 이상 안 쓴다.**

    s21에서 22자가 되어 30자 예산(=4+22+4)을 정확히 채웠다. 대회 규칙이 30자이므로
    다음 변경은 무조건 막힌다. → 파일명은 `next_zip_path`가 **변경분 + 지문**으로 만든다.
    """
    tm = describe_tokens(meta)
    head = [tm["tm"]] if "tm" in tm else []
    return "_".join(head + ["".join(v for k, v in tm.items() if k != "tm")])


def fingerprint(tok):
    """설정 토큰 전체의 4자리 지문. **충돌하지 않는 것이 유일한 요구사항이다.**

    사람이 읽는 부분(변경분)은 짧아서 두 설정이 우연히 같은 힌트를 가질 수 있다.
    지문이 그 경우를 막는다 — 파일명의 안전 기능은 여기에 있고, 앞쪽 토큰은 편의다.
    """
    import hashlib
    payload = json.dumps(tok, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:4]


HISTORY = SUBMIT / "history.json"


def load_history():
    if HISTORY.exists():
        return json.loads(HISTORY.read_text(encoding="utf-8"))
    return []


def diff_tag(prev, cur):
    """직전 제출 대비 **바뀐 토큰만** 이어 붙인다.

    SUBMISSIONS.md의 행이 이미 "s19 + δ 표에 recency 가중" 형태로 **차분**을 적는다.
    파일명도 같은 규약을 따르면 표와 이름이 서로를 설명한다.
    축이 꺼진 경우(토큰 소멸)는 힌트로 안 보이지만, 지문이 구분을 보장한다.
    """
    add = [v for k, v in cur.items() if prev.get(k) != v]
    if add:
        return add
    return ["off"] if set(prev) - set(cur) else []


def next_zip_path(meta, suffix=""):
    """`sNN_<바뀐 토큰><지문4>.zip` — **대회 규칙 30자**를 구조적으로 지킨다.

    ⚠️ s21이 정확히 30자였다(`s21_a40h188c125as30bs30k35.zip`). 전체 설정을 이름에
       넣는 방식은 축이 늘 때마다 예산을 먹어 s08/s09·s20에서 **두 번** 충돌을 냈고,
       그때마다 "안 바뀐 축을 뺀다"로 버텼다. 더 뺄 것이 없다.

    바꾼 것: 이름의 기능을 **전체 설정 표기 → 구분과 추적**으로 좁혔다.
      · 앞쪽 토큰 = 직전 제출 대비 **바뀐 것만** (사람이 읽는 힌트)
      · 뒤 4자리  = 전체 설정 지문 (충돌 방지 — 안전 기능은 여기 있다)
      · 전체 설정 = `SUBMISSIONS.md` 표 + `model/meta.json` + `submit/history.json`

    번호는 **이미 쓴 번호의 최댓값 + 1**이다. 개수로 세면 파일을 지우거나 이름을
    바꿨을 때 같은 번호가 다시 나와 기존 제출물을 덮어쓴다.
    """
    used = [0]
    for p in ROOT.glob("s[0-9][0-9]_*.zip"):
        head = p.stem.split("_")[0][1:]
        if head.isdigit():
            used.append(int(head))
    # ⚠️ 디스크의 zip만 세면 **다른 사람이 만든 제출을 못 본다.** s26에서 실제로 났다 —
    #    혜진 `p_a60.zip`이 26번으로 표에 올라와 있는데 글롭(`sNN_*.zip`)에 안 걸려
    #    26이 재발급됐다. 표가 팀 전체의 번호대장이므로 거기 쓰인 번호도 합친다.
    sub = ROOT / "SUBMISSIONS.md"
    if sub.exists():
        used += [int(m.group(1)) for m
                 in ROW_RE.finditer(sub.read_text(encoding="utf-8"))]
    n = max(used) + 1

    tok = describe_tokens(meta)
    hist = load_history()
    prev = hist[-1]["tokens"] if hist else {}
    fp = fingerprint(tok)

    # ⚠️ **직전 제출과 설정이 완전히 같으면 멈춘다.** s20에서 실제로 났다 —
    #    §7.61이 δ의 β를 바꿨는데 그 축이 `MUTE`에 있어서 태그가 s19와 문자 단위로
    #    같게 나왔다. 경고 주석을 직접 달아놓고도 지나쳤고 눈으로 잡았다.
    #    **주석은 사람이 읽어야 작동하고 assert는 안 읽어도 작동한다.**
    #    (같은 설정을 의도적으로 재현할 때는 suffix로 구분한다 — s04가 s01을 재현한 경우.)
    if prev and prev == tok and not suffix:
        raise SystemExit("\n".join([
            f"설정이 직전 제출(s{hist[-1]['n']:02d})과 **완전히 같다**.",
            "  → 이번에 바꾼 것이 describe_tokens()에 안 들어가 있다.",
            "  → 같은 설정을 의도적으로 다시 내는 것이면 suffix= 를 줄 것."]))

    # 힌트는 **토큰 단위로** 들어가는 만큼만 넣는다. 글자로 자르면 `as30`이 `as3`이
    # 되어 오히려 오해를 만든다 — 잘린 토큰보다 없는 토큰이 낫다.
    # 30자는 대회 규칙이라 협상 대상이 아니고, 구분은 지문이 이미 보장한다.
    room = MAX_NAME - len(f"s{n:02d}_{fp}{suffix}.zip")
    hint, dropped = "", 0
    for t in diff_tag(prev, tok):
        if len(hint) + len(t) <= room:
            hint += t
        else:
            dropped += 1
    if dropped:
        print(f"  ⚠️ 30자 예산으로 변경분 토큰 {dropped}개를 이름에서 생략했다"
              f" (전체는 submit/history.json).")
    name = f"s{n:02d}_{hint}{fp}{suffix}.zip"
    assert len(name) <= MAX_NAME, name
    return ROOT / name


def record_history(zip_path, meta):
    """빌드된 제출물의 **전체 설정 토큰**을 남긴다.

    이것이 있어야 다음 빌드가 "무엇이 바뀌었나"를 계산할 수 있다.
    파일명에서 전체 설정을 뺀 대가로 생긴 기록 의무다.
    """
    tok = describe_tokens(meta)
    hist = load_history()
    n = int(zip_path.stem.split("_")[0][1:])
    hist = [h for h in hist if h["n"] != n] + [
        {"n": n, "file": zip_path.name, "tokens": tok,
         "fingerprint": fingerprint(tok), "full_tag": describe(meta)}]
    hist.sort(key=lambda h: h["n"])
    HISTORY.write_text(json.dumps(hist, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  이력 기록: submit/history.json (s{n:02d}, 지문 {fingerprint(tok)})")


def log_submission(zip_path, meta, pred_mean, elapsed):
    """제출 이력을 한 줄 추가한다. LB 점수는 받은 뒤 사람이 채운다."""
    path = ROOT / "SUBMISSIONS.md"
    if not path.exists():
        path.write_text(
            "# 제출 이력\n\n"
            "`LB` 열은 리더보드 점수를 받은 뒤 직접 채운다.\n"
            "**같은 이름으로 제출하지 말 것** — 어느 설정의 점수인지 구분이 안 된다.\n\n"
            "| # | 파일 | Trackman | prior | 피처 | 홀드아웃 BSS(oracle) | LB |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n",
            encoding="utf-8")
    # 구성 칸은 파일명 태그와 같은 정보를 담아야 한다. 예전엔 '+MLP'만 붙여서
    # 멤버 수와 alpha가 사라졌고, 표만 봐서는 s08과 s09를 구분할 수 없었다.
    cfg = str(meta["n_features"])
    if meta.get("use_mlp"):
        cfg += f"+MLP{len(meta.get('mlp_members', []))}(a{meta.get('mlp_alpha')})"
    # 사후 로짓 보정 축. s11에서는 이 줄이 없어서 표에 플래툰이 안 찍혔고 손으로 채웠다 —
    # `describe()`와 같은 종류의 누락이다. 배포에 들어간 것은 전부 여기 나와야 한다.
    NICE = {"hand": "플래툰", "count4": "카운트"}
    for name, beta in (meta.get("ctx_axes") or {}).items():
        if beta:
            cfg += f"+{NICE.get(name, name)} β{beta}"
    # 보조 헤드 λ. describe()·log_submission() 둘 다 고쳐야 한다 — 이 누락이 세 번째다
    # (s11 플래툰, s12 카운트, s13 보조헤드). 배포에 넣은 것은 **전부** 여기 나와야 한다.
    if meta.get("mlp_aux_lambda"):
        AXN = {"middle": "몰림", "reverse": "반대", "ball": "볼", "strike": "스트라이크"}
        ax = [AXN.get(m, m) for m in (meta.get("mlp_aux_modes") or [])]
        cfg += (f"+보조헤드 λ{meta['mlp_aux_lambda']}"
                + (f"({'·'.join(ax)})" if ax else ""))
    # 오염 보정 축(§7.49)과 시즌 진행분(§7.55). **s14는 이 줄이 없어서 손으로 채웠다** —
    # s11 플래툰 · s12 카운트 · s13 보조헤드에 이은 네 번째·다섯 번째 누락이다.
    # ⚠️ 배포에 새 축을 넣으면 `describe()`와 여기 **둘 다** 고친다.
    if meta.get("contam_axes"):
        CN = {"p": "투수", "b": "타자"}
        cfg += "+F오염보정(" + "·".join(CN.get(c, c) for c in meta["contam_axes"]) + ")"
    NA = {"gap_success": "투수", "gap_bsuccess": "타자"}
    for name, beta in (meta.get("asof_axes") or {}).items():
        if beta:
            cfg += f"+시즌진행분 {NA.get(name, name)} β{beta}"
    MN = {"r3_ent": "레퍼토리 균형도"}
    for name, beta in (meta.get("mix_axes") or {}).items():
        if beta:
            cfg += f"+{MN.get(name, name)} β{beta}"
    # 행 단위 F 레벨 δ_F (§7.69)
    if float(meta.get("gt_f_delta") or 0.0) * float(meta.get("gt_f_beta") or 0.0):
        cfg += f"+F레벨 δ{meta['gt_f_delta']} β{meta['gt_f_beta']}"
    # init_score (§7.109). **여섯 번째 같은 누락이다** — 위 주석들이 다섯 번을 세고 있다.
    # 레벨은 soften 뒤 가중 양성률로 만든다(그게 아니면 소프트라벨과 이중교정).
    if meta.get("use_init_score"):
        cfg += "+init_score(시즌×gt, soften후 레벨)"
    row = (f"| {zip_path.stem.split('_')[0][1:]} | `{zip_path.name}` | "
           f"{meta.get('trackman_mode','none')} | {meta['target_prior_2025']} | "
           f"{cfg} | | |")

    # ⚠️ append하면 표 아래 본문에 줄이 붙는다. **표의 마지막 행 뒤에 끼워 넣는다.**
    # ⚠️ "'| ' 뒤 2글자가 숫자"만 보면 **본문의 다른 표에 속는다.** s26에서 실제로 났다 —
    #    fold 표의 `| 2024 | 0.60 | ...` 가 표의 마지막 행으로 잡혀 그 아래 끼어들었다.
    #    제출 표의 행은 **두 번째 칸이 백틱으로 시작하는 파일명**이다. 그걸 요구한다.
    lines = path.read_text(encoding="utf-8").split("\n")
    last = max((i for i, l in enumerate(lines)
                if ROW_RE.match(l)), default=None)
    if last is None:
        print(f"  ⚠️ {path.name}에서 표를 못 찾았다. 수동으로 추가할 것:\n    {row}")
        return
    lines.insert(last + 1, row)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  이력 추가: {path.name} (표 {last+1}번째 행 뒤)")


if __name__ == "__main__":
    main()
